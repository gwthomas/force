from copy import deepcopy

import torch

from force.alg.agent import BufferedAgent
from force.config import Choice
from force.nn import Optimizer
from force.nn.loss import NAMED_LOSS_FUNCTIONS
from force.nn.util import get_device, torchify, batch_map, update_ema
from force.nn.models.value_functions import QFunctionEnsemble
from force.policies import PolicyMode
from force.util import pymean, prefix_dict_keys, stats_dict


class BufferedActorCritic(BufferedAgent):
    """Base class for actor-critic algorithms that employ replay buffers.
    (e.g. SAC, TD3, SOP, REDQ, IQL)
    The critic is an ensemble of Q functions; the actor is specified by a subclass.
    The actor and/or critic may have target networks which are updated via an
    exponential moving average toward the current parameters, as in DDPG.
    """
    class Config(BufferedAgent.Config):
        actor_optimizer = Optimizer.Config
        critic = QFunctionEnsemble.Config
        critic_optimizer = Optimizer.Config
        critic_loss_criterion = Choice(NAMED_LOSS_FUNCTIONS.keys(), default='MSE')
        target_update_period = 1
        target_update_rate = 0.005

    def __init__(self, cfg, env_info, actor,
                 use_target_actor=False, use_target_critic=True,
                 replay_buffer=None, device=None):
        assert cfg.target_update_period >= 1
        assert 0 < cfg.target_update_rate <= 1
        super().__init__(
            cfg, env_info,
            replay_buffer=replay_buffer, device=device
        )

        # Actor provided by argument; critic created here
        self.actor = actor.to(self.device)
        self.critic = QFunctionEnsemble(cfg.critic, env_info.observation_shape, env_info.action_shape).to(self.device)

        # Optimizers
        self.actor_optimizer = Optimizer(cfg.actor_optimizer, self.actor.parameters())
        self.critic_optimizer = Optimizer(cfg.critic_optimizer, self.critic.parameters())

        # Target networks
        if use_target_actor:
            self.target_actor = deepcopy(actor)
            self.target_actor.set_requires_grad(False)
        else:
            self.target_actor = None
        if use_target_critic:
            self.target_critic = deepcopy(self.critic)
            self.target_critic.set_requires_grad(False)
        else:
            self.target_critic = None

        # Loss function for Q regression
        self.critic_loss_criterion = NAMED_LOSS_FUNCTIONS[cfg.critic_loss_criterion]()

    def act(self, obs, mode: PolicyMode):
        return self.actor.act(obs, mode)

    def update_target_networks(self):
        update_rate = self.cfg.target_update_rate
        if self.target_critic is not None:
            update_ema(self.target_critic, self.critic, update_rate)
        if self.target_actor is not None:
            update_ema(self.target_actor, self.actor, update_rate)

    def compute_target_value(self, obs):
        actions = self.actor.act(obs, mode=PolicyMode.EXPLORE)
        return self.target_critic([obs, actions], which='min')

    def compute_targets(self, batch: dict):
        with torch.no_grad():
            next_values = self.compute_target_value(batch['next_observations'])
        return batch['rewards'] + self.cfg.discount * (~batch['terminals']).float() * next_values

    def compute_critic_loss(self, batch: dict):
        # Compute targets (labels) for Q regression
        targets = self.compute_targets(batch)
        self.train_diagnostics['target_critic'].append(targets.mean().item())

        # Get current Q value estimates
        qs = self.critic([batch['observations'], batch['actions']], which='all')

        # Compute loss for each model in ensemble
        return pymean([
            self.critic_loss_criterion(qs[:, i], targets)
            for i in range(self.critic.num_models)
        ])

    def update_critic(self, batch: dict):
        critic_loss = self.compute_critic_loss(batch)
        self.train_diagnostics['critic_loss'].append(critic_loss.item())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.item()

    def compute_actor_loss(self, obs):
        actions = self.actor.act(obs, mode=PolicyMode.EVAL)
        return -self.critic([obs, actions], which='min').mean()

    def update_actor(self, observations):
        actor_loss = self.compute_actor_loss(observations)
        self.train_diagnostics['actor_loss'].append(actor_loss.item())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item()

    def update_with_minibatch(self, batch: dict, counters: dict):
        self.update_critic(batch)
        self.update_actor(batch['observations'])
        if counters['updates'] % self.cfg.target_update_period == 0:
            self.update_target_networks()

    def additional_diagnostics(self):
        info = super().additional_diagnostics()
        observations = self.replay_buffer.get('observations').to(self.device)
        actions = self.replay_buffer.get('actions').to(self.device)
        rewards = self.replay_buffer.get('rewards').to(self.device)
        terminals = self.replay_buffer.get('terminals').to(self.device)

        # Compute Q(s,a) for all (s,a) in buffer
        with torch.no_grad():
             q_buffer = batch_map(
                lambda s, a: self.critic([s, a], which='random'),
                [observations, actions]
             )
             info[f'q_buffer_mean'] = q_buffer.mean()

        # Stats of Q error (i.e., |Q(s,a) - r(s,a)|) for terminal transitions
        if terminals.any():
            terminal_errors = torch.abs(q_buffer[terminals] - rewards[terminals])
            info.update(prefix_dict_keys('terminal_Q_error', stats_dict(terminal_errors)))

        return info
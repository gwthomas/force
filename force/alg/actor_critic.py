from copy import deepcopy

import torch

from force.alg.agent import BufferedAgent
from force.config import Field, Choice
from force.env.util import space_shape
from force.nn import ConfigurableModule, Optimizer
from force.nn.loss import NAMED_LOSS_FUNCTIONS
from force.nn.util import get_device, torchify, batch_map, update_ema
from force.nn.models.value_functions import QFunctionEnsemble
from force.util import dict_get, pymean


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

    def __init__(self, cfg, obs_space, act_space, actor,
                 use_actor_target=False, use_critic_target=True,
                 device=None):
        assert cfg.target_update_period >= 1
        assert 0 < cfg.target_update_rate <= 1
        super().__init__(cfg, obs_space, act_space, device=device)
        obs_shape = space_shape(obs_space)
        act_shape = space_shape(act_space)

        # Actor provided by argument; critic created here
        self.actor = actor.to(self.device)
        self.critic = QFunctionEnsemble(cfg.critic, obs_shape, act_shape).to(self.device)

        # Optimizers
        self.actor_optimizer = Optimizer(cfg.actor_optimizer, self.actor.parameters())
        self.critic_optimizer = Optimizer(cfg.critic_optimizer, self.critic.parameters())

        # Target networks
        if use_actor_target:
            self.actor_target = deepcopy(actor)
            self.actor_target.set_requires_grad(False)
        else:
            self.actor_target = None
        if use_critic_target:
            self.critic_target = deepcopy(self.critic)
            self.critic_target.set_requires_grad(False)
        else:
            self.critic_target = None

        # Loss function for Q regression
        self.critic_loss_criterion = NAMED_LOSS_FUNCTIONS[cfg.critic_loss_criterion]()

    def act(self, obs, eval):
        return self.actor.act(obs, eval)

    def update_target_networks(self):
        update_rate = self.cfg.target_update_rate
        if self.critic_target is not None:
            update_ema(self.critic_target, self.critic, update_rate)
        if self.actor_target is not None:
            update_ema(self.actor_target, self.actor, update_rate)

    def compute_value(self, obs, use_target_network: bool):
        if use_target_network:
            assert self.critic_target is not None
            critic = self.critic_target
        else:
            critic = self.critic
        actions = self.act(obs, eval=False)
        return critic([obs, actions], which='min')

    def compute_targets(self, batch: dict):
        with torch.no_grad():
            next_values = self.compute_value(
                batch['next_observations'],
                use_target_network=(self.critic_target is not None)
            )
        return batch['rewards'] + self.cfg.discount * (~batch['terminals']).float() * next_values

    def compute_critic_loss(self, batch: dict):
        # Compute targets (labels) for Q regression
        targets = self.compute_targets(batch)
        self.train_diagnostics['critic_target'].append(targets.mean().item())

        # Get current Q value estimates
        qs = self.critic([batch['observations'], batch['actions']], which='all')

        # Compute loss for each model in ensemble
        losses = [
            self.critic_loss_criterion(qs[:, i], targets)
            for i in range(self.critic.num_models)
        ]
        return pymean(losses)

    def update_critic(self, batch: dict):
        critic_loss = self.compute_critic_loss(batch)
        self.train_diagnostics['critic_loss'].append(critic_loss.item())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.item()

    def compute_actor_loss(self, obs):
        actions = self.actor.act(obs, eval=True)
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
            info['terminal_Q_error_mean'] = terminal_errors.mean()
            info['terminal_Q_error_max'] = terminal_errors.max()

        return info
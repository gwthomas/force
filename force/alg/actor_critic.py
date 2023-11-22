from abc import abstractmethod
from copy import deepcopy

import torch

from force.alg.base import Agent
from force.config import Field
from force.nn import ConfigurableModule, Optimizer
from force.nn.loss import NAMED_LOSS_FUNCTIONS
from force.nn.util import get_device, update_ema
from force.sampling import ReplayBuffer
from force.util import batch_map, dict_get_several, pymean


class ActorCritic(Agent):
    class Config(Agent.Config):
        actor_optimizer = Optimizer.Config()
        critic_optimizer = Optimizer.Config()
        critic_loss_criterion = 'MSE'
        target_update_rate = Field(0.005, check=lambda x: 0 < x <= 1)

    def __init__(self, cfg, obs_space, act_space, actor, critic,
                 use_actor_target=False, use_critic_target=True,
                 device=None):
        super().__init__(cfg, obs_space, act_space)
        self._device = device = get_device(device)
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.actor_optimizer = Optimizer(cfg.actor_optimizer, actor.parameters())
        self.critic_optimizer = Optimizer(cfg.critic_optimizer, critic.parameters())
        self.critic_loss_criterion = NAMED_LOSS_FUNCTIONS[cfg.critic_loss_criterion]()

        if use_actor_target:
            self.actor_target = deepcopy(actor)
            self.actor_target.set_requires_grad(False)
        else:
            self.actor_target = None

        if use_critic_target:
            self.critic_target = deepcopy(critic)
            self.critic_target.set_requires_grad(False)
        else:
            self.critic_target = None

    def act(self, states, eval):
        return self.actor.act(states, eval)

    def sync_target_networks(self):
        if self.critic_target is not None:
            print('Syncing critic target network')
            self.critic_target.load_state_dict(self.critic.state_dict())
        if self.actor_target is not None:
            print('Syncing actor target network')
            self.actor_target.load_state_dict(self.actor.state_dict())

    def update_target_networks(self):
        update_rate = self.cfg.target_update_rate
        if self.critic_target is not None:
            update_ema(self.critic_target, self.critic, update_rate)
        if self.actor_target is not None:
            update_ema(self.actor_target, self.actor, update_rate)

    @abstractmethod
    def compute_value(self, obs, use_target_network):
        raise NotImplementedError

    def compute_critic_loss(self, batch):
        obs, actions, next_obs, rewards, terminals, _ = dict_get_several(batch, *batch.keys())
        with torch.no_grad():
            next_values = self.compute_value(next_obs, use_target_network=True)
        targets = rewards + self.cfg.discount * (~terminals).float() * next_values
        self.train_diagnostics['critic_target'].append(targets.mean().item())
        qs = self.critic([obs, actions], which='all')
        return pymean([self.critic_loss_criterion(qs[:, qi], targets)
                       for qi in range(self.critic.num_models)])

    def compute_actor_loss(self, batch):
        obs = batch['observations']
        actions = self.actor.act(obs, eval=True)
        return -self.critic([obs, actions], which='random').mean()

    def update_critic(self, batch: dict):
        critic_loss = self.compute_critic_loss(batch)
        self.train_diagnostics['critic_loss'].append(critic_loss.item())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.item()

    def update_actor(self, batch: dict):
        actor_loss = self.compute_actor_loss(batch)
        self.train_diagnostics['actor_loss'].append(actor_loss.item())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item()

    def update_with_batch(self, batch: dict, counters: dict):
        self.update_critic(batch)
        self.update_actor(batch)
        self.update_target_networks()

    def additional_diagnostics(self, buffer: ReplayBuffer):
        obs, actions, next_obs, rewards, terminals, _ = buffer.get()
        info = super().additional_diagnostics(buffer)
        with torch.no_grad():
            info[f'q_buffer'] = batch_map(
                lambda s, a: self.critic([s, a], which='random'),
                [obs, actions]
            ).mean()
        return info
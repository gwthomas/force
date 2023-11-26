from copy import deepcopy
import random

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from force.alg.actor_critic import ActorCritic
from force.alg.base import Agent
from force.config import Field, Choice
from force.env.util import space_shape
from force.nn import MLP, Optimizer
from force.nn.loss import NAMED_LOSS_FUNCTIONS
from force.nn.util import get_device, update_ema, select1_per_row
from force.policies import SquashedGaussianPolicy
from force.sampling import ReplayBuffer
from force.util import batch_map, pymean
from force.value_functions import QFunctionEnsemble, DiscreteQFunctionEnsemble


class SAC(ActorCritic):
    class Config(ActorCritic.Config):
        actor = SquashedGaussianPolicy
        critic = QFunctionEnsemble
        init_alpha = 1.0
        autotune_alpha = True
        target_entropy = Field(float, required=False)
        use_log_alpha_loss = True
        deterministic_backup = False

    def __init__(self, cfg, obs_space, act_space,
                 actor=None, critic=None, device=None):
        obs_shape = space_shape(obs_space)
        act_shape = space_shape(act_space)
        act_dim = act_shape[0]
        if actor is None:
            actor = SquashedGaussianPolicy(cfg.actor, obs_shape, act_dim)
        if critic is None:
            critic = QFunctionEnsemble(cfg.critic, obs_shape, act_shape)
        device = get_device(device)

        super().__init__(cfg, obs_space, act_space, actor, critic,
                         use_actor_target=False, use_critic_target=True,
                         device=device)

        self.log_alpha = torch.tensor(cfg.init_alpha, device=device).log()
        if self.cfg.autotune_alpha:
            self.log_alpha = nn.Parameter(self.log_alpha)
            self.alpha_optimizer = Optimizer(cfg.actor_optimizer, [self.log_alpha])
            if self.cfg.target_entropy is None:
                self.cfg.target_entropy = -act_dim   # set target entropy to -dim(A)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def compute_value(self, obs, use_target_network):
        critic = self.critic_target if use_target_network else self.critic
        distr = self.actor.distr(obs)
        action = distr.rsample()
        value = critic([obs, action], which='min')
        if not self.cfg.deterministic_backup:
            value = value - self.alpha * distr.log_prob(action)
        return value

    def compute_actor_loss(self, batch: dict):
        obs = batch['observations']
        distr = self.actor.distr(obs)
        action = distr.rsample()
        log_prob = distr.log_prob(action)
        actor_q = self.critic([obs, action], which='random')
        alpha = self.alpha
        actor_loss = torch.mean(alpha.detach() * log_prob - actor_q)
        if self.cfg.autotune_alpha:
            multiplier = self.log_alpha if self.cfg.use_log_alpha_loss else alpha
            alpha_loss = -multiplier * torch.mean(log_prob.detach() + self.cfg.target_entropy)
            return [actor_loss, alpha_loss]
        else:
            return [actor_loss]

    def update_actor(self, batch: dict):
        losses = self.compute_actor_loss(batch)
        self.train_diagnostics['actor_loss'].append(losses[0].item())
        optimizers = [self.actor_optimizer]
        if self.cfg.autotune_alpha:
            optimizers.append(self.alpha_optimizer)
        assert len(losses) == len(optimizers)
        for loss, optimizer in zip(losses, optimizers):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def additional_diagnostics(self, data: ReplayBuffer) -> dict:
        info = super().additional_diagnostics(data)
        info['alpha'] = self.alpha.detach()

        data = data.get(as_dict=True)
        obs = data['observations']
        def surprise(o):
            distr = self.actor.distr(o)
            sample = distr.sample()
            return -distr.log_prob(sample)
        with torch.no_grad():
            info['entropy'] = batch_map(surprise, obs).mean()
        return info


def is_positive(x):
    return x > 0

# Given two matrices (of the same shape), computes inner products row-wise
def dot_rows(a, b):
    assert a.shape == b.shape and len(a.shape) == 2
    return (a * b).sum(1)

def probs_and_logs(qs, alpha):
    log_probs = F.log_softmax(qs / alpha, dim=1)
    probs = torch.exp(log_probs)
    return probs, log_probs

def soft_value(qs, alpha):
    probs, log_probs = probs_and_logs(qs, alpha)
    return dot_rows(probs, qs - alpha * log_probs)

# Discrete version of SAC
# We represent the policy implicitly rather than having a separate network
class DSAC(Agent):
    class Config(Agent.Config):
        critic = DiscreteQFunctionEnsemble
        loss_type = Choice(NAMED_LOSS_FUNCTIONS.keys(), default='MSE')
        critic_optimizer = Optimizer
        target_update_rate = Field(0.005, check=is_positive)
        alpha_optimizer = Optimizer
        init_alpha = 1.0
        autotune_alpha = True
        target_entropy = Field(float, required=False, check=is_positive)
        use_log_alpha_loss = False #True
        deterministic_backup = False

    def __init__(self, cfg, obs_space, act_space,
                 device=None):
        assert isinstance(act_space, gym.spaces.Discrete)
        obs_shape = space_shape(obs_space)
        num_actions = act_space.n
        device = get_device(device)
        super().__init__(cfg, obs_space, act_space)

        self.critic = DiscreteQFunctionEnsemble(cfg.critic, obs_shape, num_actions).to(device)
        self.critic_target = deepcopy(self.critic)
        self.loss_fn = NAMED_LOSS_FUNCTIONS[cfg.loss_type]()
        self.critic_optimizer = Optimizer(cfg.critic_optimizer, self.critic.parameters())

        self.log_alpha = torch.tensor(self.cfg.init_alpha, device=device).log()
        if self.cfg.autotune_alpha:
            self.log_alpha = nn.Parameter(self.log_alpha)
            self.alpha_optimizer = Optimizer(cfg.alpha_optimizer, [self.log_alpha])
            if self.cfg.target_entropy is None:
                max_entropy = torch.tensor(num_actions).float().log()
                self.cfg.target_entropy = 0.5 * max_entropy

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, eval):
        with torch.no_grad():
            qvals = self.critic(obs)
            qvals = qvals.mean(dim=1)  # average across ensemble
        if eval:
            # Greedy action
            return qvals.argmax(-1)
        else:
            # Sample from Boltzmann distribution
            return torch.distributions.Categorical(logits=qvals/self.alpha).sample()

    def update_with_batch(self, batch, counters):
        obs, actions, next_obs, rewards, terminals, _ = batch.values()
        actions = actions.long()

        # Compute targets
        with torch.no_grad():
            next_values = soft_value(self.critic(next_obs).min(dim=1).values, self.alpha)
            targets = rewards + (~terminals).float() * self.cfg.discount * next_values

        all_qs = self.critic(obs)
        critic_losses = []
        for i in range(all_qs.shape[1]):
            action_qs = select1_per_row(all_qs[:,i,:], actions)
            critic_losses.append(self.loss_fn(action_qs, targets))
        critic_loss = pymean(critic_losses)
        self.train_diagnostics['critic_loss'].append(critic_loss.item())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update critic target network
        update_ema(self.critic_target, self.critic, self.cfg.target_update_rate)

        # Update alpha (if applicable)
        if self.cfg.autotune_alpha:
            with torch.no_grad():
                probs, log_probs = probs_and_logs(all_qs.mean(dim=1), self.alpha)
            multiplier = self.log_alpha if self.cfg.use_log_alpha_loss else self.alpha
            alpha_loss = -dot_rows(probs, multiplier * (log_probs + self.cfg.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

    def additional_diagnostics(self, buffer):
        obs, actions, next_obs, rewards, terminals, _ = buffer.get()
        alpha = self.alpha.detach()

        info = super().additional_diagnostics(buffer)
        info['alpha'] = alpha
        with torch.no_grad():
            info['value'] = batch_map(
                lambda o: soft_value(self.critic(o).mean(dim=1), alpha),
                obs
            ).mean()
        return info
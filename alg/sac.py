import math
import torch
from torch import nn
import torch.nn.functional as F

from ..policy import SquashedGaussianPolicy
from ..torch_util import device, mlp, update_ema
from ..train import get_optimizer
from .actor_critic import ActorCritic
from .td3 import Critic


class SAC(ActorCritic):
    def __init__(self, state_dim, action_dim,
                 discount=0.99,
                 init_temperature=0.2,
                 actor_update_period=1,
                 tau=0.005,
                 batch_size=256,
                 actor=None, critic=None, log_alpha=None):
        if actor is None:
            actor = SquashedGaussianPolicy(mlp([state_dim, 256, 256, action_dim*2]))
        if critic is None:
            critic = Critic(state_dim, action_dim)
        if log_alpha is None:
            log_alpha = torch.tensor(math.log(init_temperature), device=device, requires_grad=True)
        super().__init__(actor, critic, use_actor_target=False, use_critic_target=True)

        self.discount = discount
        self.tau = tau
        self.actor_update_period = actor_update_period
        self.batch_size = batch_size

        self.log_alpha = log_alpha
        self.log_alpha_optimizer = get_optimizer([self.log_alpha])
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        self.total_updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update_critic(self, obs, action, next_obs, reward, done):
        if reward.dim() == 1:
            reward = reward.unsqueeze(1)
        if done.dim() == 1:
            done = done.unsqueeze(1)
        not_done = 1 - done.float()

        with torch.no_grad():
            distr = self.actor.distr(next_obs)
            next_action = distr.rsample()
            log_prob = torch.unsqueeze(distr.log_prob(next_action), 1)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + not_done * self.discount * target_V

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        update_ema(self.critic_target, self.critic, self.tau)

    def _compute_actor_grad(self, obs, maximize=False, also_alpha=True):
        dist = self.actor.distr(obs)
        action = dist.rsample()
        log_prob = torch.unsqueeze(dist.log_prob(action), 1)
        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        if maximize:
            actor_loss = -actor_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        if also_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            if maximize:
                alpha_loss = -alpha_loss
            alpha_loss.backward()

    def actor_grad(self, obs, maximize=False, include_alpha=False):
        self._compute_actor_grad(obs, maximize=maximize, also_alpha=include_alpha)
        grad = [p.grad.clone() for p in self.actor.parameters()]
        if include_alpha:
            return grad, self.log_alpha.grad.clone()
        else:
            return grad

    def update_actor_and_alpha(self, obs):
        self._compute_actor_grad(obs, also_alpha=True)
        self.actor_optimizer.step()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer):
        obs, action, next_obs, reward, done = replay_buffer.sample(self.batch_size)
        self.update_critic(obs, action, next_obs, reward, done)
        self.total_updates += 1
        if self.total_updates % self.actor_update_period == 0:
            self.update_actor_and_alpha(obs)
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from force.alg.actor_critic import ActorCritic
from force.policy import BasePolicy
from force.torch_util import device, Module, mlp, torchify, update_ema
from force.train import get_optimizer


class BoltzmannActor(BasePolicy):
    def __init__(self, logit_fn):
        self.logit_fn = logit_fn

    def probs(self, obs):
        return F.softmax(self.logit_fn(torchify(obs)), dim=1)

    def act(self, obs, eval):
        logits = self.logit_fn(torchify(obs))
        if eval:
            actions = logits.argmax(dim=1)
        else:
            actions = Categorical(logits=logits).sample()
        return actions.cpu().numpy()


class DiscreteCritic(Module):
    def __init__(self, obs_dim, n_actions, hidden_dims=(256, 256)):
        super(DiscreteCritic, self).__init__()
        self.q1 = mlp([obs_dim, *hidden_dims, n_actions])
        self.q2 = mlp([obs_dim, *hidden_dims, n_actions])

    def forward(self, obs):
        return self.q1(obs), self.q2(obs)

    def min_Qs(self, obs):
        Q1s, Q2s = self(obs)
        return torch.min(Q1s, Q2s)


# Given two matrices (of the same shape), computes inner products row-wise
def dot_rows(a, b):
    assert a.shape == b.shape and len(a.shape) == 2
    return (a * b).sum(1)

class DSAC(ActorCritic):
    def __init__(self, state_dim, n_actions,
                 discount=0.99,
                 init_temperature=0.2,
                 tau=0.005,
                 epsilon=1e-8,
                 alpha_update_period=1,
                 critic_target_update_period=1,
                 batch_size=256):
        actor = BoltzmannActor(self.actor_logits)
        critic = DiscreteCritic(state_dim, n_actions)
        super().__init__(actor, critic, use_actor_target=False, use_critic_target=True)

        self.discount = discount
        self.tau = tau
        self.epsilon = epsilon

        self.alpha_update_period = alpha_update_period
        self.critic_target_update_period = critic_target_update_period
        self.batch_size = batch_size

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = get_optimizer([self.log_alpha])
        self.target_entropy = 0.5 * np.log(n_actions)

        self.total_updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def actor_logits(self, obs):
        with torch.no_grad():
            return self.critic.min_Qs(obs) / self.alpha

    def update_critic(self, obs, action, reward, next_obs, not_done):
        with torch.no_grad():
            next_probs = self.actor.probs(next_obs)
            next_log_probs = torch.log(next_probs + self.epsilon)
            min_Qs = self.critic_target.min_Qs(next_obs)
            target_V = dot_rows(next_probs, min_Qs - self.alpha * next_log_probs)
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1s, current_Q2s = self.critic(obs)
        current_Q1 = current_Q1s.gather(1, action.unsqueeze(1)).squeeze()
        current_Q2 = current_Q2s.gather(1, action.unsqueeze(1)).squeeze()

        assert current_Q1.shape == target_Q.shape
        assert current_Q2.shape == target_Q.shape

        critic_loss = F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_alpha(self, obs):
        with torch.no_grad():
            probs = self.actor.probs(obs)
            log_probs = torch.log(probs + self.epsilon)

        alpha_loss = dot_rows(probs, self.alpha * (-log_probs - self.target_entropy)).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer):
        obs, action, next_obs, reward, done = replay_buffer.sample(self.batch_size)
        action = action.long()
        not_done = 1 - done

        self.update_critic(obs, action, reward, next_obs, not_done)
        self.total_updates += 1

        if self.total_updates % self.alpha_update_period == 0:
            pass
            # self.update_alpha(obs)

        if self.total_updates % self.critic_target_update_period == 0:
            update_ema(self.critic_target, self.critic, self.tau)
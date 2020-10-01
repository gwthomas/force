import torch
import torch.nn.functional as F

from .actor_critic import ActorCritic
from ..policy import BasePolicy
from ..torch_util import Module, mlp, update_ema

# Based on author's original implementation, see https://github.com/sfujim/TD3


class Actor(Module, BasePolicy):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = mlp([state_dim, 256, 256, action_dim])

    def forward(self, state):
        return torch.tanh(self.net(state))

    def act(self, state, eval):
        with torch.no_grad():
            return self(state)


class Critic(Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.q1 = mlp([state_dim + action_dim, 256, 256, 1])
        self.q2 = mlp([state_dim + action_dim, 256, 256, 1])

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa)


class TD3(ActorCritic):
    def __init__(
            self,
            state_dim,
            action_dim,
            batch_size=256,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            delay=2,
            actor=None, critic=None,
    ):
        if actor is None:
            actor = Actor(state_dim, action_dim)
        if critic is None:
            critic = Critic(state_dim, action_dim)
        super().__init__(actor, critic, use_actor_target=True, use_critic_target=True)

        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.delay = delay

        self.total_updates = 0

    def update_critic(self, obs, action, next_obs, reward, done):
        if reward.dim() == 1:
            reward = reward.unsqueeze(1)
        if done.dim() == 1:
            done = done.unsqueeze(1)
        not_done = 1 - done.float()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_obs) + noise
            ).clamp(-1, 1)

            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        update_ema(self.critic_target, self.critic, self.tau)

    def _compute_actor_grad(self, obs, maximize=False):
        self.actor.zero_grad()
        actor_loss = -self.critic.Q1(obs, self.actor(obs)).mean()
        if maximize:
            actor_loss = -actor_loss
        actor_loss.backward()

    def update(self, replay_buffer):
        self.update_critic(*replay_buffer.sample(self.batch_size))
        self.total_updates += 1

        # Delayed policy updates
        if self.total_updates % self.delay == 0:
            self._compute_actor_grad()
            self.actor_optimizer.step()
            update_ema(self.actor_target, self.actor, self.tau)

    def actor_grad(self, obs, maximize=False):
        self._compute_actor_grad(obs, maximize=maximize)
        return [p.grad.clone() for p in self.actor.parameters()]
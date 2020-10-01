import torch
import torch.nn as nn
import torch.nn.functional as F

from .actor_critic import ActorCritic
from ..torch_util import Module, mlp, torchify, update_ema


# Adapted from https://github.com/sfujim/BCQ

# Returns an action for a given state
class Actor(Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()
		self.net = mlp([state_dim, 400, 300, action_dim])
		self.max_action = max_action

	def forward(self, state):
		return self.max_action * torch.tanh(self.net(state))


# Returns a Q-value for given state/action pair
class Critic(Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400 + action_dim, 300)
		self.l3 = nn.Linear(300, 1)

	def forward(self, state, action):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(torch.cat([q, action], 1)))
		q = self.l3(q)
		return q


class DDPG(ActorCritic):
	def __init__(
			self,
			state_dim,
			action_dim,
			max_action,
			discount=0.99,
			tau=0.005,
	):
		actor = Actor(state_dim, action_dim, max_action)
		critic = Critic(state_dim, action_dim)
		super().__init__(actor, critic, actor_optimizer_args={'lr': 1e-4}, critic_optimizer_args={'weight_decay': 1e-2})

		self.state_dim = state_dim
		self.discount = discount
		self.tau = tau

	def update(self, replay_buffer, batch_size=100):
		state, action, next_state, reward, done = replay_buffer.sample(batch_size)
		state = torchify(state)
		action = torchify(action)
		next_state = torchify(next_state)
		reward = torchify(reward)
		not_done = torchify(1 - done)

		# Compute the target Q value
		target_Q = self.critic_target(next_state, self.actor_target(next_state))
		target_Q = reward + (not_done * self.discount * target_Q).detach()

		# Get current Q estimate
		current_Q = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actor_loss = -self.critic(state, self.actor(state)).mean()

		# Optimize the actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		update_ema(self.critic_target, self.critic, self.tau)
		update_ema(self.actor_target, self.actor, self.tau)
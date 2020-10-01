import torch
import torch.nn.functional as F
from tqdm import trange

from force.etc.cvae import CVAE
from force.torch_util import device, Module, mlp, torchify, update_ema
from force.train import get_optimizer


# Based on author's original implementation, see https://github.com/sfujim/BCQ

class Actor(Module):
	def __init__(self, state_dim, action_dim, max_action, max_perturbation=0.05):
		super(Actor, self).__init__()
		input_dim = state_dim + action_dim
		self.net = mlp([input_dim, 400, 300, action_dim])
		self.max_action = max_action
		self.max_perturbation = max_perturbation

	def forward(self, state, action):
		a = self.net(torch.cat([state, action], 1))
		a = self.max_perturbation * self.max_action * torch.tanh(a)
		return (a + action).clamp(-self.max_action, self.max_action)


class Critic(Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		input_dim = state_dim + action_dim
		self.net1 = mlp([input_dim, 400, 300, 1])
		self.net2 = mlp([input_dim, 400, 300, 1])

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		return self.net1(sa), self.net2(sa)

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)
		return self.net1(sa)


class BCQ:
	def __init__(self, state_dim, action_dim, max_action, max_perturbation=0.05):
		input_dim = state_dim + action_dim
		latent_dim = action_dim * 2

		self.actor = Actor(state_dim, action_dim, max_action, max_perturbation=max_perturbation)
		self.actor_target = Actor(state_dim, action_dim, max_action, max_perturbation=max_perturbation)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = get_optimizer(self.actor)

		self.critic = Critic(state_dim, action_dim)
		self.critic_target = Critic(state_dim, action_dim)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = get_optimizer(self.critic)

		self.vae = CVAE(state_dim, action_dim, latent_dim, max_action).to(device)
		self.vae_optimizer = get_optimizer(self.vae)

		self.max_action = max_action
		self.action_dim = action_dim

	def act(self, states, eval):
		with torch.no_grad():
			states = torchify(states.reshape(1, -1)).repeat(10, 1)
			actions = self.actor(states, self.vae.decode(states))
			q1 = self.critic.Q1(states, actions)
			ind = q1.max(0)[1]
		return actions[ind].cpu().data.numpy().flatten()

	def train(self, replay_buffer, iterations, batch_size=256, discount=0.99, tau=0.005):
		for it in trange(iterations):
			# Sample replay buffer / batch
			state, action, next_state, reward, done = replay_buffer.sample(batch_size)
			reward = reward.unsqueeze(1)
			done = (1 - done).unsqueeze(1)

			# Variational Auto-Encoder Training
			vae_loss = self.vae.loss(state, action)
			self.vae_optimizer.zero_grad()
			vae_loss.backward()
			self.vae_optimizer.step()

			# Critic Training
			with torch.no_grad():
				# Duplicate state 10 times
				# state_rep = torchify(np.repeat(next_state_np, 10, axis=0))
				state_rep = next_state.repeat_interleave(10, dim=0)
				
				# Compute value of perturbed actions sampled from the VAE
				target_Q1, target_Q2 = self.critic_target(state_rep, self.actor_target(state_rep, self.vae.decode(state_rep)))

				# Soft Clipped Double Q-learning 
				target_Q = 0.75 * torch.min(target_Q1, target_Q2) + 0.25 * torch.max(target_Q1, target_Q2)
				target_Q = target_Q.view(batch_size, -1).max(1)[0].view(-1, 1)

				target_Q = reward + done * discount * target_Q

			current_Q1, current_Q2 = self.critic(state, action)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Pertubation Model / Action Training
			sampled_actions = self.vae.decode(state)
			perturbed_actions = self.actor(state, sampled_actions)

			# Update through DPG
			actor_loss = -self.critic.Q1(state, perturbed_actions).mean()

			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update Target Networks
			update_ema(self.critic_target, self.critic, tau)
			update_ema(self.actor_target, self.actor, tau)
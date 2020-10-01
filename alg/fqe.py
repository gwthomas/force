import torch
import torch.nn as nn

from force.log import default_logger as log
from force.torch_util import Module, mlp, torchify
from force.train import epochal_training, get_optimizer
from force.util import batch_map


class Critic(Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim+action_dim, *hidden_dim, 1])

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], 1)).squeeze()


class FQE:
    def __init__(self, policy, obs_dim, action_dim, discount=0.99, inner_epochs=10):
        self.policy = policy
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discount = discount
        self.inner_epochs = inner_epochs
        self.iterations_completed = 0
        self.criterion = nn.SmoothL1Loss()
        self.reset_critic()

    def reset_critic(self):
        self.critic = Critic(self.obs_dim, self.action_dim)
        self.critic_optimizer = get_optimizer(self.critic)

    def value(self, obs):
        with torch.no_grad():
            action = self.policy.act(obs, eval=False)
            values = self.critic(obs, action)
            return values

    def compute_critic_target(self, reward, not_done, next_obs):
        return reward + not_done * self.discount * self.value(next_obs)

    def compute_critic_loss(self, obs, action, target):
        return self.criterion(self.critic(obs, action), target)

    def train(self, replay_buffer, iterations=1, map_batch_size=1000, verbose=True):
        obs, action, next_obs, reward, done = replay_buffer.get()
        not_done = 1 - done
        for _ in range(iterations):
            if verbose:
                log.message('Computing targets')
            target = batch_map(self.compute_critic_target, [reward, not_done, next_obs], batch_size=map_batch_size)

            if verbose:
                log.message('Updating critic')
            epochal_training(self.compute_critic_loss, self.critic_optimizer, [obs, action, target],
                             epochs=self.inner_epochs, verbose=verbose)

            self.iterations_completed += 1
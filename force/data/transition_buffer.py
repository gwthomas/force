import gymnasium as gym
import torch

from force.data import CircularDataset, TensorDataset
from force.env.util import space_dim
from force.nn.util import batch_iterator
from force.util import compute_returns


class TransitionBuffer(CircularDataset):
    def __init__(self, observation_space, action_space, capacity,
                 device=None):
        self._observation_space = observation_space
        self._action_space = action_space

        obs_dim = space_dim(observation_space)
        obs_shape = torch.Size([obs_dim])
        scalar_shape = torch.Size([])
        discrete_actions = isinstance(action_space, gym.spaces.Discrete)
        if discrete_actions:
            action_dtype = torch.int
            action_shape = scalar_shape
        else:
            action_dtype = torch.float
            action_shape = torch.Size([space_dim(action_space)])

        components = {
            'observations': (torch.float, obs_shape),
            'actions': (action_dtype, action_shape),
            'next_observations': (torch.float, obs_shape),
            'rewards': (torch.float, scalar_shape),
            'terminals': (torch.bool, scalar_shape),
            'truncateds': (torch.bool, scalar_shape)
        }
        super().__init__(components, capacity, device)

    def separate_into_trajectories(self):
        data = self.get(as_dict=True)
        trajectories = []
        traj_returns = []
        if not data['terminals'].any():
            for traj in batch_iterator(data, self.T, shuffle=False):
                rewards = traj['rewards']
                traj['rtgs'] = compute_returns(rewards, self.cfg.dt.discount)
                traj['truncateds'] = torch.zeros(self.T, dtype=bool)
                traj['truncateds'][-1] = True
                traj = TensorDataset(traj, copy=True, device=self.data_device)
                traj.share_memory()
                trajectories.append(traj)
                traj_returns.append(rewards.sum().item())
        else:
            breakpoint()

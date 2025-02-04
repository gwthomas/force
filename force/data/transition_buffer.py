import gymnasium as gym
import torch

from force.data import CircularDataset, TensorDataset
from force.nn.util import batch_iterator
from force.util import compute_returns


class TransitionBuffer(CircularDataset):
    def __init__(self, env_info, capacity,
                 device=None, additional_components=None):
        self.env_info = env_info

        obs_shape = env_info.observation_shape
        act_shape = env_info.action_shape
        act_dtype = env_info.action_dtype
        scalar_shape = torch.Size([])

        components = {
            'observations': (torch.float, obs_shape),
            'actions': (act_dtype, act_shape),
            'next_observations': (torch.float, obs_shape),
            'rewards': (torch.float, scalar_shape),
            'terminals': (torch.bool, scalar_shape),
            'truncateds': (torch.bool, scalar_shape)
        }
        if additional_components is not None:
            assert isinstance(additional_components, dict)
            for k, v in additional_components.items():
                assert k not in components
                components[k] = v
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
            # to-do
            raise NotImplementedError

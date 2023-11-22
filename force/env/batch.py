import torch

from force.env.util import get_max_episode_steps
from force.nn.util import get_device, torchify


class BatchedEnv:
    def __init__(self, envs, device=None):
        self.envs = envs
        self.device = device = get_device(device)

        proto_env = envs[0]
        self.max_episode_steps = get_max_episode_steps(proto_env)
        # All envs must have same max steps
        for env in envs:
            assert get_max_episode_steps(env) == self.max_episode_steps

        self.observation_space = proto_env.observation_space
        self.action_space = proto_env.action_space

        obs_dim = self.observation_space.shape[0]
        self._last_obs = torch.zeros(self.num_envs, obs_dim, device=device)

    @property
    def num_envs(self):
        return len(self.envs)

    def partial_reset(self, indices):
        initial_obs = torch.stack([
            torchify(self.envs[index].reset()[0]) for index in indices
        ])
        self._last_obs[indices] = initial_obs
        return initial_obs

    def reset(self):
        return self.partial_reset(torch.arange(self.num_envs))

    def step(self, actions):
        next_observations, rewards, terminateds, truncateds, infos = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_observations.append(torchify(next_obs, device=self.device))
            rewards.append(float(reward))
            terminateds.append(bool(terminated))
            truncateds.append(bool(truncated))
            infos.append(info)
        next_observations = torch.stack(next_observations)
        rewards = torch.tensor(rewards, device=self.device)
        terminateds = torch.tensor(terminateds, device=self.device)
        truncateds = torch.tensor(truncateds, device=self.device)
        self._last_obs.copy_(next_observations)
        return next_observations, rewards, terminateds, truncateds, infos
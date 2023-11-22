import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import RescaleAction, TimeLimit

import numpy as np
import torch

from force.nn.util import get_device, numpyify, torchify


class TorchWrapper(gym.Wrapper):
    def __init__(self, env, device=None):
        super().__init__(env)
        self.device = get_device(device)

    def reset(self):
        return torchify(self.env.reset(), device=self.device)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(numpyify(action))
        return torchify(observation, device=self.device), float(reward), terminated, truncated, info


def get_gym_env(env_name,
                max_episode_steps=None,
                rescale_action=True,
                wrap_torch=True,
                device=None,
                seed=None):
    env = gym.make(env_name, max_episode_steps=max_episode_steps)
    if rescale_action and isinstance(env.action_space, Box) and not (
            np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)):
        env = RescaleAction(env, -1, 1)
    if wrap_torch:
        env = TorchWrapper(env, device=device)
    env.action_space.seed(seed)
    return env


def isbox(space):
    return isinstance(space, Box)

def isdiscrete(space):
    return isinstance(space, Discrete)


def space_dim(space):
    if isbox(space):
        return int(np.prod(space.shape))
    elif isdiscrete(space):
        return space.n
    else:
        raise ValueError(f'Unknown space {space}')

def space_shape(space):
    return torch.Size([space_dim(space)])


def env_dims(env):
    return (space_dim(env.observation_space), space_dim(env.action_space))

def env_shapes(env):
    return (space_shape(env.observation_space), space_shape(env.action_space))


def get_max_episode_steps(env):
    if isinstance(env, TimeLimit):
        return env._max_episode_steps
    elif isinstance(env, gym.Wrapper):
        return get_max_episode_steps(env.env)
    elif hasattr(env, 'max_episode_steps'):
        return env.max_episode_steps
    else:
        raise ValueError('env does not have max_episode_steps')


def get_done(env):
    if hasattr(env.__class__, 'done'):
        return env.__class__.done
    else:
        return get_done(env.env)
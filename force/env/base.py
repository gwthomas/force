from abc import ABC, abstractmethod
from dataclasses import dataclass

import gymnasium as gym
from gymnasium.spaces import Space, Box, Discrete
from gymnasium.envs.registration import EnvSpec
import numpy as np
import torch

from force.nn.shape import Shape
from force.nn.util import get_device, torchify, numpyify
from force.types import Action


# Space-related utility functions
def is_box(space) -> bool:
    return isinstance(space, Box)

def is_discrete(space) -> bool:
    return isinstance(space, Discrete)

def is_standard_box(space) -> bool:
    return is_box(space) and np.all(space.low == -1.) and np.all(space.high == 1.)    

def space_shape(space) -> Shape:
    if is_box(space):
        return torch.Size(space.shape)
    elif is_discrete(space):
        return torch.Size([])
    else:
        raise ValueError(f'Unknown space {space}')

def space_dtype(space) -> torch.dtype:
    if is_box(space):
        return torch.float32
    elif is_discrete(space):
        return torch.int32
    else:
        raise ValueError(f'Unknown space {space}')


@dataclass
class EnvInfo:
    """Convenience class to package all information about an env
    that is shared across instances."""
    observation_space: Space
    action_space: Space
    spec: EnvSpec

    @property
    def observation_shape(self) -> Shape:
        return space_shape(self.observation_space)
    
    @property
    def action_shape(self) -> Shape:
        return space_shape(self.action_space)
    
    @property
    def action_dtype(self) -> torch.dtype:
        return space_dtype(self.action_space)
    
    @property
    def horizon(self) -> int:
        return self.spec.max_episode_steps


class BaseEnv(ABC):
    def __init__(self, info: EnvInfo, device=None):
        self._info = info
        self._device = get_device(device)

    @property
    def info(self):
        return self._info
    
    @property
    def observation_space(self):
        return self.info.observation_space
    
    @property
    def action_space(self):
        return self.info.action_space

    @property
    def device(self):
        return self._device

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        raise NotImplementedError


class GymEnv(BaseEnv):
    def __init__(self, id: str, device=None, make_kwargs=None, wrappers=None):
        if make_kwargs is None:
            make_kwargs = {}
        
        gym_env = gym.make(id, **make_kwargs)
        if wrappers is not None:
            for wrapper in wrappers:
                gym_env = wrapper(gym_env)
        self.gym_env = gym_env

        env_info = EnvInfo(
            observation_space=gym_env.observation_space,
            action_space=gym_env.action_space,
            spec=gym_env.spec
        )
        super().__init__(env_info, device)

    def reset(self):
        obs, info = self.gym_env.reset()
        obs = torchify(obs, device=self.device)
        return obs, info

    def step(self, action: Action):
        np_action = numpyify(action)
        assert self.gym_env.action_space.contains(np_action)
        next_obs, reward, terminated, truncated, info = self.gym_env.step(np_action)
        next_obs = torchify(next_obs, device=self.device)
        reward = float(reward)
        return next_obs, reward, terminated, truncated, info
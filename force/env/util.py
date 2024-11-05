import random

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import torch
import numpy as np

from force.config import BaseConfig, Field
from force.defaults import MAX_SEED
from force.env.torch_wrapper import TorchWrapper, TorchVectorWrapper


class EnvConfig(BaseConfig):
    name = str
    max_episode_steps = Field(int, required=False)
    render_mode = Field(int, required=False)


class VectorEnvConfig(EnvConfig):
    num_envs = 1
    asynchronous = False


def get_env(env_cfg: EnvConfig,
            device=None,
            seed=None):
    make_kwargs = {
        'max_episode_steps': env_cfg.max_episode_steps,
        'render_mode': env_cfg.render_mode
    }
    env = gym.make(env_cfg.name, **make_kwargs)
    is_vec = isinstance(env_cfg, VectorEnvConfig)
    if is_vec:
        env_spec = env.spec
        env = gym.vector.make(env_cfg.name, num_envs=env_cfg.num_envs,
                              asynchronous=env_cfg.asynchronous, **make_kwargs)
        env.spec = env_spec
    else:
        env = gym.make(env_cfg.name, **make_kwargs)

    # If a seed is not provided, one will be randomly generated (for the sake
    # of variety across env instances) according to the already-seeded random
    # module's RNG (for the sake of reproducibility)
    if seed is None:
        seed = random.randrange(MAX_SEED)
        env.reset(seed=seed)
        env.action_space.seed(seed)

    if is_vec:
        env = TorchVectorWrapper(env, device=device)
    else:
        env = TorchWrapper(env, device=device)
    return env


# Space-related utility functions
def is_box(space):
    return isinstance(space, Box)

def is_discrete(space):
    return isinstance(space, Discrete)

def is_standard_box(space):
    return is_box(space) and np.all(space.low == -1.) and np.all(space.high == 1.)

def space_dim(space):
    if is_box(space):
        return int(np.prod(space.shape))
    elif is_discrete(space):
        return space.n
    else:
        raise ValueError(f'Unknown space {space}')

def space_shape(space):
    return torch.Size([space_dim(space)])


def env_dims(env):
    return (space_dim(env.observation_space), space_dim(env.action_space))

def env_shapes(env):
    return (space_shape(env.observation_space), space_shape(env.action_space))
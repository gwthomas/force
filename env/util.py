import gym
from gym.spaces import Box, Discrete
from gym.wrappers import RescaleAction, TimeLimit

import numpy as np

from .mujoco import MUJOCO_ENVS
from .whynot import WHYNOT_ENVS
from .torch_wrapper import TorchWrapper

MAX_EPISODE_STEPS = {
    'HIV': 400
}

for mjenv in MUJOCO_ENVS:
    MAX_EPISODE_STEPS[mjenv] = 1000


def time_limit(env, max_episode_steps):
    from gym.wrappers import TimeLimit
    try:
        get_max_episode_steps(env)
    except:
        return TimeLimit(env, max_episode_steps=max_episode_steps)


def get_env(env_name, seed=None, rescale_action=True, wrap_torch=True):
    if env_name in MUJOCO_ENVS:
        env = MUJOCO_ENVS[env_name]()
    elif env_name in WHYNOT_ENVS:
        env = WHYNOT_ENVS[env_name]()
    else:
        env = gym.make(env_name)

    if rescale_action and isinstance(env.action_space, Box):
        env = RescaleAction(env, -1, 1)

    if env_name in MAX_EPISODE_STEPS:
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS[env_name])

    if wrap_torch:
        env = TorchWrapper(env)

    env.seed(seed)
    return env


def isbox(space):
    return isinstance(space, Box)

def isdiscrete(space):
    return isinstance(space, Discrete)


def space_dim(space):
    if isbox(space):
        return np.prod(space.shape)
    elif isdiscrete(space):
        return space.n
    else:
        raise ValueError(f'Unknown space {space}')


def env_dims(env):
    return (space_dim(env.observation_space), space_dim(env.action_space))


def get_max_episode_steps(env):
    if hasattr(env, '_max_episode_steps'):
        return env._max_episode_steps
    elif hasattr(env, 'env'):
        return get_max_episode_steps(env.env)
    else:
        raise ValueError('env does not have _max_episode_steps')
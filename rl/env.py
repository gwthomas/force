import gym
import numpy as np
import tensorflow as tf

import gtml.config as cfg
from gtml.util.memory import Memory
from gtml.util.misc import attrwise_cat


def integral_dimensionality(space):
    if isinstance(space, gym.spaces.Discrete):
        return space.n
    elif isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape))

def placeholder_for_space(space, extra_dims, name=None):
    if isinstance(space, gym.spaces.Discrete):
        type = cfg.INT_T
        shape = []
    elif isinstance(space, gym.spaces.Box):
        type = cfg.FLOAT_T
        shape = list(space.shape)
    else:
        raise RuntimeError

    shape = [None]*extra_dims + shape
    return tf.placeholder(type, shape, name=name)

class Environment:
    def __init__(self, name, discount=0.99, preprocess=None, history=10):
        if preprocess is None:
            preprocess = lambda raw, _: raw

        self.gym_env = gym.make(name)
        self.discount = discount
        self.preprocess = preprocess
        self.observation_history = Memory(history)

    @property
    def observation_space(self):
        return self.gym_env.observation_space

    @property
    def action_space(self):
        return self.gym_env.action_space

    def reset(self):
        self.observation_history.clear()
        raw_observation = self.gym_env.reset()
        self.observation_history.add(raw_observation)
        return self.preprocess(raw_observation, self)

    def step(self, action):
        raw_observation, reward, done, info = self.gym_env.step(action)
        self.observation_history.add(raw_observation)
        return self.preprocess(raw_observation, self), reward, done, info

    def render(self):
        self.gym_env.render()

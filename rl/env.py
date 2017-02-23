import gym
import numpy as np
import theano.tensor as T

import gtml.config as cfg
from gtml.util.memory import Memory
from gtml.util.misc import attrwise_cat

# Determines how many dimensions an observation or action space has
# If it's a discrete space, this is the number of elements
# If it's a box space, this is the shape of the box
def dimensionality(space):
    if isinstance(space, gym.spaces.Discrete):
        return space.n
    elif isinstance(space, gym.spaces.Box):
        return space.shape

def integral_dimensionality(space):
    return int(np.prod(dimensionality(space)))

def var_for_space(space):
    if isinstance(space, gym.spaces.Discrete):
        return T.iscalar()
    elif isinstance(space, gym.spaces.Box):
        dim = len(dimensionality(space))
        return T.TensorType(cfg.FLOAT_T, (False,)*dim)()

def vector_var_for_space(space):
    if isinstance(space, gym.spaces.Discrete):
        return T.ivector()
    elif isinstance(space, gym.spaces.Box):
        dim = len(dimensionality(space))
        return T.TensorType(cfg.FLOAT_T, (False,)*(dim+1))()


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

import gym
import numpy as np
from scipy.misc import imresize
import tensorflow as tf
import types

from gtml.common.memory import Memory
from gtml.common.util import luminance
from gtml.defaults import FLOAT_T, INT_T


def integral_dimensionality(space):
    if isinstance(space, gym.spaces.Discrete):
        return space.n
    elif isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape))

def placeholder_for_space(space, extra_dims=1, name=None):
    if isinstance(space, gym.spaces.Discrete):
        type = INT_T
        shape = []
    elif isinstance(space, gym.spaces.Box):
        type = FLOAT_T
        shape = list(space.shape)
    else:
        raise RuntimeError

    shape = [None]*extra_dims + shape
    return tf.placeholder(type, shape, name=name)


class Environment:
    def __init__(self, name, discount=0.99, history=10):
        self.name = name
        self.discount = discount
        self.gym_env = gym.make(name)
        self.raw_obs_history = Memory(history)

    @property
    def observation_space(self):
        return self.gym_env.observation_space

    @property
    def action_space(self):
        return self.gym_env.action_space

    def reset(self):
        self.raw_obs_history.clear()
        raw_observation = self.gym_env.reset()
        self.raw_obs_history.add(raw_observation)
        return self.preprocess(raw_observation)

    def step(self, action):
        raw_observation, reward, done, info = self.gym_env.step(action)
        self.raw_obs_history.add(raw_observation)
        return self.preprocess(raw_observation), reward, done, info

    def render(self):
        self.gym_env.render()

    def preprocess(self, raw_observation):
        return raw_observation

    def create_observation_placeholder(self, extra_dims=1, name=None):
        return placeholder_for_space(self.observation_space, extra_dims=extra_dims, name=name)

    def create_action_placeholder(self, extra_dims=1, name=None):
        return placeholder_for_space(self.action_space, extra_dims=extra_dims, name=name)


# Implements preprocessing method described in the paper by Mnih, et al.
class AtariEnvironment:
    def __init__(self, name, discount=0.99, history=10, m=4, size=(84,84)):
        Environment.__init__(self, name, discount=discount, history=history)
        self.m = m
        self.size = size

    def preprocess(self, raw_observation):
        recent_raw = self.raw_obs_history.recent(m)

        # Make sure there are enough frames (duplicate latest if not)
        latest = recent_raw
        while len(recent_raw) < m+1:
            # This assignment is safe; a new list is created
            recent_raw = [latest] + raw_observations

        # Calculate luminance and resize
        recent_frames = []
        for i in range(m):
            maxed = np.maximum(recent_raw[-(1+i)], recent_raw[-(i+2)])
            luma = luminance(maxed)
            resized = imresize(luma, size).astype('float32')
            recent_frames.append(resized)

        # Stack and normalize pixel values
        return np.stack(recent_frames) / 255.0

    def create_observation_placeholder(self, extra_dims=1, name=None):
        shape = [None]*extra_dims + [self.size, self.size, self.m]
        return tf.placeholder(FLOAT_T, shape, name=name)

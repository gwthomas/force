import gym
from scipy.misc import imresize
import torch

from gtml.memory import Memory
from gtml.util import luminance
from gtml.defaults import DISCOUNT

def integral_dimensionality(space):
    if isinstance(space, gym.spaces.Discrete):
        return space.n
    elif isinstance(space, gym.spaces.Box):
        return int(torch.prod(torch.Tensor(space.shape)))


class Environment:
    def __init__(self, name, discount=DISCOUNT, history=10):
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

    @property
    def discrete_actions(self):
        return isinstance(self.action_space, gym.spaces.Discrete)

    def reset(self):
        self.raw_obs_history.clear()
        raw_observation = torch.from_numpy(self.gym_env.reset()).float()
        self.raw_obs_history.add(raw_observation)
        return self.preprocess(raw_observation)

    def step(self, action):
        raw_observation, reward, done, info = self.gym_env.step(action)
        raw_observation = torch.from_numpy(raw_observation).float()
        self.raw_obs_history.add(raw_observation)
        return self.preprocess(raw_observation), reward, done, info

    def render(self):
        self.gym_env.render()

    def preprocess(self, raw_observation):
        return raw_observation


# Implements preprocessing method described in the paper by Mnih, et al.
class AtariEnvironment(Environment):
    def __init__(self, name, discount=0.99, m=4, size=(84,84)):
        Environment.__init__(self, name, discount=discount, history=m)
        self.m = m
        self.size = size

    def preprocess(self, raw_observation):
        recent_raw = self.raw_obs_history.recent(self.m)

        # Make sure there are enough frames (duplicate latest if not)
        latest = recent_raw[-1]
        while len(recent_raw) < self.m + 1:
            recent_raw.append(latest)

        # Calculate luminance and resize
        recent_frames = []
        for i in range(self.m):
            maxed = torch.max(recent_raw[-(i+1)], recent_raw[-(i+2)])
            luma = luminance(maxed)
            resized = torch.Tensor(imresize(luma, self.size))
            recent_frames.append(resized)

        # Stack and normalize pixel values
        return torch.stack(recent_frames) / 255.0

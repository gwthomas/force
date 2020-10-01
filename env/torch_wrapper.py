from gym import Wrapper

from force.torch_util import torchify, numpyify


class TorchWrapper(Wrapper):
    def reset(self):
        return torchify(self.env.reset())

    def step(self, action):
        observation, reward, done, info = self.env.step(numpyify(action))
        return torchify(observation), reward, done, info
import numpy as np
import tensorflow as tf

from gtml.nn.network import Container
from gtml.nn.opt import SupervisedLearning, mean_squared_error
from gtml.rl.core import discounted_returns

class ValueFunction:
    def __init__(self, env):
        self.env = env

    def fit(self, episodes):
        raise NotImplementedError

    def __call__(self, observations):
        raise NotImplementedError


class ZeroValueFunction(ValueFunction):
    def fit(self, episodes, itrs):
        pass

    def __call__(self, observations):
        return np.zeros(len(observations))


class ParametricValueFunction(ValueFunction, Container):
    def __init__(self, env, implementation):
        super().__init__(env)
        Container.__init__(self, implementation)

    def fit(self, episodes, itrs):
        X, Y = [], []
        for episode in episodes:
            X.extend(episode.observations)
            Y.extend(discounted_returns(episode.rewards, self.env.discount))
        raise NotImplementedError
        # self.opt.run(np.array(X), np.column_stack([Y]), itrs=itrs, batchsize=100)

    def __call__(self, observations):
        return self.implementation(observations).flatten()

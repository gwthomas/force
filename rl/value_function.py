import numpy as np
import tensorflow as tf

from gtml.nn.network import Container

class ValueFunction:
    def __init__(self, env):
        self.env = env

    def __call__(self, observations):
        raise NotImplementedError


class ZeroValueFunction(ValueFunction):
    def __call__(self, observations):
        return np.zeros(len(observations))


class ParametricValueFunction(ValueFunction, Container):
    def __init__(self, env, implementation):
        super().__init__(env)
        Container.__init__(self, implementation)

    def __call__(self, observations):
        return self.implementation(observations).flatten()

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import force.util as util

class Policy:
    def act(self, observations):
        raise NotImplementedError

    def act1(self, observation):
        return self.act(tf.expand_dims(observation, axis=0))[0]


class ParametricPolicy(Policy):
    def __init__(self, net):
        self.net = net

    def clone(self):
        return self.__class__(util.clone_model(self.net))


class DeterministicPolicy(ParametricPolicy):
    def act(self, observations):
        return self.net(observations)


class StochasticPolicy(ParametricPolicy):
    def action_distributions(self, observations):
        raise NotImplementedError

    def act(self, observations):
        return self.action_distributions(observations).sample().numpy()


class CategoricalPolicy(StochasticPolicy):
    def action_distributions(self, observations):
        logits = self.net(observations)
        return tfd.Categorical(logits=logits)


class GaussianPolicy(StochasticPolicy):
    def action_distributions(self, observations):
        mean, std = self.net(observations)
        return tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)

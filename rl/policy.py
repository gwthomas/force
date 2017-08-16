import numpy as np
import tensorflow as tf; tfdists = tf.contrib.distributions

from gtml.common.tf import get_sess


class Policy:
    def act(self, observations, sess=None):
        raise NotImplementedError


class TFPolicy(Policy):
    def __init__(self, observations_in, actions):
        self.observations_in = observations_in
        self.actions = actions

    def act(self, observations, sess=None):
        sess = get_sess(sess)
        return sess.run(self.actions, feed_dict={self.observations_in: observations})


class StochasticPolicy(TFPolicy):
    def __init__(self, observations_in, pdist):
        self.pdist = pdist
        actions = pdist.sample()
        TFPolicy.__init__(self, observations_in, actions)

    def prob(self, actions):
        return self.pdist.prob(actions)

    def entropy(self):
        return self.pdist.entropy()


class SoftmaxPolicy(StochasticPolicy):
    def __init__(self, observations_in, logits):
        self.logits = logits
        pdist = tfdists.Categorical(logits=logits)
        StochasticPolicy.__init__(self, observations_in, pdist)

import numpy as np
import tensorflow as tf

from gtml.nn.network import Container
from gtml.util.misc import add_dim


class Policy(object):
    def __init__(self, env):
        self.env = env

    def get_action(self, observation):
        raise NotImplementedError


class ParametricPolicy(Policy, Container):
    def __init__(self, env, implementation):
        super().__init__(env)
        Container.__init__(self, implementation)

    def get_action(self, observation):
        return self.get_actions(add_dim(observation))[0]

    def get_actions(self, observation):
        raise NotImplementedError


class DirectPolicy(ParametricPolicy):
    stochastic = False

    def get_actions(self, observations):
        return self.implementation(observations)


class ArgmaxPolicy(ParametricPolicy):
    stochastic = False

    def get_actions(self, observations):
        output = self.implementation(observations)
        return np.argmax(output)


class MultinomialPolicy(ParametricPolicy):
    stochastic = True

    def get_actions(self, observations):
        output = self.implementation(observations)
        return np.array([np.random.choice(len(row), p=row) for row in output])

    def get_log_probs_var(self, actions_var):
        all_log_probs = T.log(self.implementation.get_output_var())
        return all_log_probs[T.arange(actions_var.shape[0]),actions_var]

    def get_entropy_var(self):
        output = self.get_output_var()
        return -T.sum(output * T.log(output))


# Sort of a meta-policy. Takes another policy as input
class EpsilonGreedyPolicy(Policy):
    stochastic = True

    def __init__(self, policy, epsilon):
        super(EpsilonGreedyPolicy, self).__init__(policy.env)
        self.policy = policy
        self.epsilon = epsilon

    def get_action(self, observation):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.policy.get_action(observation)

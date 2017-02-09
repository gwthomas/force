from policy import Policy
import numpy as np


# Sort of a meta-policy. Takes another policy as input
class EpsilonGreedyPolicy(Policy):
    def __init__(self, policy, epsilon):
        super(EpsilonGreedyPolicy, self).__init__(policy.env)
        self.policy = policy
        self.epsilon = epsilon

    def get_action(self, observation):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.policy.get_action(observation)


# Take the greedy action w.r.t. the given Q-function
class QGreedyPolicy(Policy):
    def __init__(self, q):
        super(QGreedyPolicy, self).__init__(q.env)
        self.q = q

    def get_action(self, observation):
        return self.q.best_action(observation)

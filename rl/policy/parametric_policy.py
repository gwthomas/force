from .policy import Policy
from gtml.nn.container import Container
from gtml.util.misc import add_dim
import numpy as np

# Skeleton for function approximation. We assume that the implementation's call
# operator produces either
#   1) distibutions over actions
#   or
#   2) value(s) directly representing an action (e.g. for continuous action spaces)
# for several observations at a time
class ParametricPolicy(Policy, Container):
    INTERPRETATIONS = ('stochastic', 'greedy-stochastic', 'direct')

    def __init__(self, env, implementation, interpretation):
        assert interpretation in ParametricPolicy.INTERPRETATIONS
        super(ParametricPolicy, self).__init__(env)
        Container.__init__(self, implementation)
        self.interpretation = interpretation

    def get_action(self, observation):
        output = self.implementation(add_dim(observation))[0]
        if self.interpretation == 'stochastic':
            return np.random.choice(len(output), p=output)
        elif self.interpretation == 'greedy-stochastic':
            return np.argmax(output)
        elif self.interpretation == 'direct':
            return output

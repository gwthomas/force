import numpy as np
import tensorflow as tf

from gtml.nn.network import Network

class Policy(Network):
    def __init__(self, env, policy, name=None):
        self.env = env
        self.policy = policy
        self.observations_in = policy.get_orig_input()
        self.names = [policy.name]
        super().__init__([policy], name if name is not None else policy.name)

    def act(self, observations, sess=None):
        results = self.eval(self.names, {self.observations_in: observations}, sess=sess)
        assert '_actions' not in results # reserved
        results['_actions'] = results[self.policy.name]
        return results


# Special policy that also spits out value function estimates
class ActorCritic(Policy):
    def __init__(self, env, actor, critic, name):
        self.actor = actor
        self.critic = critic
        super().__init__(env, actor)
        self.names.append(critic.name)
        # Network init already got called but it needs both outputs
        Network.__init__(self, [actor, critic], name)

    def act(self, observations, sess=None):
        results = self.eval(self.names, {self.observations_in: observations}, sess=sess)
        assert '_actions' not in results # reserved
        assert '_values' not in results  # reserved
        results['_actions'] = results[self.actor.name].flatten()
        results['_values'] = results[self.critic.name].flatten()
        return results

    def critique(self, observations, sess=None):
        results = self.eval([self.critic.name], {self.observations_in: observations}, sess=sess)
        return results[self.critic.name].flatten()

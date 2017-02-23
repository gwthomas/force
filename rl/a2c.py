from multiprocessing import Process, Queue, cpu_count
import lasagne
import numpy as np
import theano
import theano.tensor as T

from gtml.nn.network import Collection, Container
from gtml.nn.opt import Optimizer, squared_error
from gtml.rl.core import Episode, partial_rollout, rollouts, discounted_returns
from gtml.rl.env import vector_var_for_space
from gtml.util.misc import conflattenate, clip_by_global_norm


class ActorCritic(Collection):
    def __init__(self, actor, critic, name):
        self.actor = actor
        self.critic = critic
        super().__init__([actor, critic], name)


class A2C(Optimizer):
    def __init__(self, setup_fn, load=False, update_fn=lasagne.updates.adam, reg_value_fit=0.25, reg_entropy=0.01):
        ac = setup_fn()
        if load:
            ac.load_params()
        policy, value_fn = ac.actor, ac.critic
        env = policy.env

        observations_var = policy.get_input_var()
        actions_var = vector_var_for_space(env.action_space)
        log_probs_var = policy.get_log_probs_var(actions_var)
        returns_var = T.fvector()
        advantages_var = T.fvector()
        policy_loss_var = -T.sum(log_probs_var * advantages_var)
        if reg_entropy != 0:
            policy_loss_var = policy_loss_var - reg_entropy * policy.get_entropy_var()
        value_fn_loss_var = squared_error(value_fn.get_output_var().flatten(), returns_var)
        loss_var = policy_loss_var + reg_value_fit * value_fn_loss_var
        input_vars = [observations_var, actions_var, returns_var, advantages_var]
        updates = update_fn(loss_var, ac.get_param_vars())

        super().__init__(input_vars, updates)
        self.ac = ac

    def run(self, Tmax, tmax=20, render=False):
        policy, value_fn = self.ac.actor, self.ac.critic
        env = policy.env
        episode = Episode()
        T = 0
        while T < Tmax:
            # Act for a bit
            steps = partial_rollout(policy, episode, tmax, render=render)
            T += steps
            if episode.done:
                R = 0
                observations = episode.observations[-steps:]
            else:
                R = value_fn([episode.latest_observation()])[0]
                observations = episode.observations[-(steps+1):-1]
            actions = episode.actions[-steps:]
            rewards = episode.rewards[-steps:]
            returns = np.zeros(steps)
            for i in range(steps-1, -1, -1):
                R = rewards[i] + env.discount * R
                returns[i] = R

            advantages = returns - value_fn(observations)
            self.step(observations, actions, returns, advantages)
            self.ac.save_params()

            if episode.done:
                print(episode.discounted_return)
                episode = Episode()

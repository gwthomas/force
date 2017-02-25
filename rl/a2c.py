import numpy as np
import tensorflow as tf

import gtml.config as cfg
from gtml.nn.network import Network
from gtml.rl.core import Episode, partial_rollout, rollouts, discounted_returns
from gtml.rl.env import placeholder_for_space
from gtml.util.tf import flatten, squared_error

class ActorCritic(Network):
    def __init__(self, actor, critic, name):
        self.actor = actor
        self.critic = critic
        super().__init__([actor.implementation, critic.implementation], name)


class A2C:
    def __init__(self, setup_fn, optimizer=tf.train.AdamOptimizer(), reg_value_fit=0.25, reg_entropy=0.01):
        ac = setup_fn()
        policy, value_fn = ac.actor, ac.critic
        env = policy.env

        observations_tf = policy.get_orig_input()
        actions_tf = placeholder_for_space(env.action_space, 1, name='actions')
        n_tf = tf.placeholder(cfg.INT_T, shape=[], name='n')
        log_probs_tf = policy.get_log_probs_var(actions_tf, n_tf)

        returns_tf = tf.placeholder(cfg.FLOAT_T, shape=[None], name='returns')
        advantages_tf = tf.placeholder(cfg.FLOAT_T, shape=[None], name='advantages')
        policy_loss_tf = -tf.reduce_sum(log_probs_tf * advantages_tf)
        if reg_entropy != 0:
            policy_loss_tf = policy_loss_tf - reg_entropy * policy.get_entropy()
        value_fn_loss_tf = squared_error(flatten(value_fn.get_output()), returns_tf)
        loss_tf = policy_loss_tf + reg_value_fit * value_fn_loss_tf
        self.train = optimizer.minimize(loss_tf)
        self.ac = ac
        self.observations_tf = observations_tf
        self.actions_tf = actions_tf
        self.returns_tf = returns_tf
        self.advantages_tf = advantages_tf
        self.n_tf = n_tf

    def run(self, Tmax, tmax=20, render=False):
        policy, value_fn = self.ac.actor, self.ac.critic
        env = policy.env
        episode = Episode()
        T = 0
        while T < Tmax:
            # Act for a bit
            steps = partial_rollout(policy, episode, tmax, render=render)
            value_predictions = value_fn(episode.observations)
            T += steps
            if episode.done:
                R = 0
                observations = episode.observations[-steps:]
            else:
                R = value_predictions[-1]
                observations = episode.observations[-(steps+1):-1]
            actions = episode.actions[-steps:]
            rewards = episode.rewards[-steps:]
            returns = np.zeros(steps)
            for i in range(steps-1, -1, -1):
                R = rewards[i] + env.discount * R
                returns[i] = R

            advantages = returns - value_predictions[:steps]
            self.train.run(feed_dict={
                    self.observations_tf: observations,
                    self.actions_tf: actions,
                    self.returns_tf: returns,
                    self.advantages_tf: advantages,
                    self.n_tf: len(actions)
            })

            if episode.done:
                print(episode.discounted_return)
                episode = Episode()

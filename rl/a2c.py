import numpy as np
import tensorflow as tf

import gtml.config as cfg
from gtml.nn.network import Network
from gtml.rl.core import Episode
from gtml.rl.env import placeholder_for_space
from gtml.util.tf import flatten, get_sess, squared_error


class A2C:
    def __init__(self, setup_fn, optimizer=tf.train.AdamOptimizer(), reg_value_fit=0.25, reg_entropy=0.01):
        ac = setup_fn()
        policy = ac.policy
        env = ac.env

        observations_in = ac.observations_in
        actions_in = placeholder_for_space(env.action_space, 1, name='actions')
        n_in = tf.placeholder(cfg.INT_T, shape=[], name='n')
        log_probs = ac.policy.get_log_probs(actions_in, n_in)

        returns_in = tf.placeholder(cfg.FLOAT_T, shape=[None], name='returns')
        advantages_in = tf.placeholder(cfg.FLOAT_T, shape=[None], name='advantages')
        actor_loss = -tf.reduce_sum(log_probs * advantages_in)
        if reg_entropy != 0:
            actor_loss = actor_loss - reg_entropy * policy.get_entropy()
        critic_loss = squared_error(flatten(ac.critic.get_output()), returns_in)
        loss = actor_loss + reg_value_fit * critic_loss
        self.actor_loss = actor_loss
        self.critic_loss = critic_loss
        self.loss = loss
        self.train = optimizer.minimize(loss)
        self.ac = ac
        self.observations_in = observations_in
        self.actions_in = actions_in
        self.returns_in = returns_in
        self.advantages_in = advantages_in
        self.n_in = n_in

    def run(self, Tmax, tmax=20, render=False, sess=None):
        sess = get_sess(sess)
        env = self.ac.env
        episode = Episode()
        T = 0
        while T < Tmax:
            steps = episode.run(self.ac, tmax, render=render, sess=sess)
            values = episode.policy_outputs['_values'][-steps:]
            T += steps
            if episode.done:
                R = 0
                observations = episode.observations[-steps:]
            else:
                R = self.ac.critique([episode.observations[-1]])
                observations = episode.observations[-(steps+1):-1]
            actions = episode.actions[-steps:]
            rewards = episode.rewards[-steps:]
            returns = np.zeros(steps)
            for i in range(steps-1, -1, -1):
                R = rewards[i] + env.discount * R
                returns[i] = R

            advantages = returns - values
            results = sess.run([self.actor_loss, self.critic_loss, self.train], {
                    self.observations_in: observations,
                    self.actions_in: actions,
                    self.returns_in: returns,
                    self.advantages_in: advantages,
                    self.n_in: len(actions)
            })
            # print(results[:-1])

            if episode.done:
                print(episode.discounted_return)
                episode = Episode()

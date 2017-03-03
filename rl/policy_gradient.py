import numpy as np
import tensorflow as tf

import gtml.config as cfg
from gtml.nn.network import Network
from gtml.rl.core import Episode, rollouts, discounted_returns
from gtml.rl.env import placeholder_for_space
from gtml.util.misc import attrwise_cat
from gtml.util.tf import flatten, get_sess, squared_error


def concat_episodes(episodes):
    return attrwise_cat(episodes, ['observations', 'actions', 'rewards'])


class PolicyGradientMethod:
    def __init__(self, ac, batchsize=1, optimizer=tf.train.AdamOptimizer(),
            reg_critic=0.25, reg_entropy=0.01):
        env = ac.env

        observations_in = ac.observations_in
        actions_in = placeholder_for_space(env.action_space, 1, name='actions')
        n_in = tf.placeholder(cfg.INT_T, shape=[], name='n')
        log_probs = ac.policy.get_log_probs(actions_in, n_in)

        returns_in = tf.placeholder(cfg.FLOAT_T, shape=[None], name='returns')
        advantages_in = tf.placeholder(cfg.FLOAT_T, shape=[None], name='advantages')
        policy_loss = -tf.reduce_sum(log_probs * advantages_in)
        if reg_entropy != 0:
            policy_loss = policy_loss - reg_entropy * ac.actor.get_entropy()
        critic_loss = squared_error(flatten(ac.critic.get_output()), returns_in)
        loss = policy_loss + reg_critic * critic_loss
        self.policy_loss = policy_loss
        self.critic_loss = critic_loss
        self.loss = loss
        self.train = optimizer.minimize(policy_loss)
        self.fit = optimizer.minimize(critic_loss)
        self.ac = ac
        self.observations_in = observations_in
        self.actions_in = actions_in
        self.returns_in = returns_in
        self.advantages_in = advantages_in
        self.batchsize = batchsize

    def run(self, num_episodes=1, render=False, sess=None):
        sess = get_sess(sess)
        env = self.ac.env
        num_updates = num_episodes // self.batchsize
        for _ in range(num_updates):
            episodes = rollouts(self.ac, self.batchsize, render=render)
            print([episode.discounted_return for episode in episodes])
            observations, actions, rewards = concat_episodes(episodes)
            returns, advantages = [], []
            for episode in episodes:
                Rt = discounted_returns(episode.rewards, env.discount)
                returns.extend(Rt)
                advantages.extend(Rt - self.ac.critique(episode.observations))
            results = sess.run(self.train, {
                    self.observations_in: observations,
                    self.actions_in: actions,
                    self.returns_in: returns,
                    self.advantages_in: advantages
            })
            # print(results[:-1])
            for _ in range(10):
                sess.run(self.fit, {
                        self.observations_in: observations,
                        self.actions_in: actions,
                        self.returns_in: returns,
                        self.advantages_in: advantages
                })

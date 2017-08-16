import numpy as np
import tensorflow as tf

from gtml.common.tf import get_sess
from gtml.defaults import BATCHSIZE, FLOAT_T, GAE_LAMBDA, OPTIMIZER
from gtml.nn.variable import VariableManager, get_default_variable_manager
from gtml.rl.util import estimate_advantages_and_value_targets
from gtml.train.minimizer import Minimizer


class ProximalPolicyOptimization(Minimizer):
    def __init__(self, env, factory, optimizer=OPTIMIZER, batchsize=BATCHSIZE,
            variable_manager=get_default_variable_manager(), epsilon=0.2, T=128,
            n_actors=1, epochs_per_update=3, c_vf=1.0, c_ent=0.01, gae_lambda=GAE_LAMBDA):
        self.env = env
        self.variable_manager = variable_manager
        self.epsilon = epsilon
        self.T = T
        self.n_actors = n_actors
        self.epochs_per_update = epochs_per_update
        self.c_vf = c_vf
        self.c_ent = c_ent
        self.gae_lambda = gae_lambda

        self.observations_in = env.create_observation_placeholder()
        self.actions_in = env.create_action_placeholder()
        self.advantages_in = tf.placeholder(FLOAT_T, [None])
        self.value_targets_in = tf.placeholder(FLOAT_T, [None])
        with tf.variable_scope('ppo'):
            self.policy, self.value_fn = factory(env, self.observations_in, self.variable_manager)
            self.variable_manager.use_sync_group('ppo_old')
            self.policy_old, self.value_fn_old = factory(env, self.observations_in, self.variable_manager)

        r = self.policy.prob(self.actions_in) / self.policy_old.prob(self.actions_in)
        L_clip = tf.minimum(r * self.advantages_in, tf.clip_by_value(r, 1 - epsilon, 1 + epsilon) * self.advantages_in)
        L_vf = tf.reduce_sum((self.value_fn - self.value_targets_in)**2)
        L_ent = self.policy.entropy()
        L = L_clip - self.c_vf * L_vf + self.c_ent * L_ent
        loss = -tf.reduce_mean(L)
        Minimizer.__init__(self, loss, optimizer=optimizer, batchsize=batchsize)

    def run(self, engine, n_iterations=1, sess=None):
        assert engine.env is self.env
        sess = get_sess(sess)
        episode = engine.new_episode()
        for itr in range(n_iterations):
            observations, actions, advantages, value_targets = [], [], [], []
            for actor in range(self.n_actors):
                steps_taken = episode.run(self.policy, self.T)
                new_observations = episode.observations[-(steps_taken+1):]
                new_values = sess.run(self.value_fn, {self.observations_in: new_observations})
                new_actions = episode.actions[-steps_taken:]
                new_rewards = episode.rewards[-steps_taken:]
                done = episode.done
                new_advantages, new_value_targets = estimate_advantages_and_value_targets(
                        new_observations, new_actions, new_rewards, done, new_values,
                        gamma=self.env.discount, lam=self.gae_lambda
                )

                observations.extend(new_observations)
                actions.extend(new_actions)
                advantages.extend(new_advantages)
                value_targets.extend(new_value_targets)

                if done:
                    episode = engine.new_episode()

            feeds = {
                self.observations_in: np.array(observations[:-1]),
                self.actions_in: np.array(actions),
                self.advantages_in: np.array(advantages),
                self.value_targets_in: np.array(value_targets)
            }
            for _ in range(self.epochs_per_update):
                self.epoch(feeds, sess=sess)

            self.variable_manager.sync('ppo_old')

import copy

import tensorflow as tf
from tensorflow.data import Dataset

from force.callbacks import CallbackManager
from force.constants import DEFAULT_BATCH_SIZE, DEFAULT_DISCOUNT, DEFAULT_GAE_LAMBDA
from force.rl.sampling import ParallelSampler
from force.train import EpochalMinimizer


class ProximalPolicyOptimization(CallbackManager):
    def __init__(self, env, policy, value_fn, optimizer,
                 clip_eps=0.2, T=128, n_actors=10,
                 c_vf=1.0, c_ent=0.01,
                 batch_size=DEFAULT_BATCH_SIZE, n_epochs=10,
                 discount=DEFAULT_DISCOUNT, gae_lambda=DEFAULT_GAE_LAMBDA):
        self.policy = policy
        self.old_policy = policy.clone()
        self.value_fn = value_fn
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.T = T
        self.c_vf = c_vf
        self.c_ent = c_ent
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.discount = discount
        self.gae_lambda = gae_lambda

        sampler = ParallelSampler(env, n_actors, {
            'discount': discount,
            'gae_lambda': gae_lambda
        })
        sampler.start()
        sampler.send('policy', (policy.__class__, policy.net.__class__, policy.net.get_config()))
        sampler.send('value_fn', (value_fn.__class__, value_fn.get_config()))
        self.sampler = sampler

        all_params = policy.net.weights + value_fn.weights
        self.train = EpochalMinimizer(self.compute_loss, all_params, optimizer, None)

        CallbackManager.__init__(self)

    def compute_loss(self, inputs):
        observations, actions, advantages, value_targets = inputs
        action_distributions = self.policy.action_distributions(observations)
        action_distributions_old = self.old_policy.action_distributions(observations)
        r = action_distributions.prob(actions) / action_distributions_old.prob(actions)
        r_clipped = tf.clip_by_value(r, 1 - self.clip_eps, 1 + self.clip_eps)
        L_clip = tf.minimum(r * advantages, r_clipped * advantages)
        L_vf = tf.reduce_sum((self.value_fn(observations) - value_targets)**2)
        L_ent = action_distributions.entropy()
        L = L_clip - self.c_vf * L_vf + self.c_ent * L_ent
        return -tf.reduce_mean(L)

    def run(self, n_iterations=1):
        for itr in range(n_iterations):
            samples = self.sampler.partial_rollout(max_steps=self.T)

            # Using zip instead of from_tensor_slices because the latter needs
            # all tensors to have the same type, and actions may be integers
            self.train.dataset = Dataset.zip((
                    Dataset.from_tensor_slices(samples['observations']),
                    Dataset.from_tensor_slices(samples['actions']),
                    Dataset.from_tensor_slices(samples['advantages']),
                    Dataset.from_tensor_slices(samples['value_targets'])
            )).batch(self.batch_size)

            print('Training...')
            self.train.run(self.n_epochs)
            self.run_callbacks('post-iteration')
            self.old_policy.net.set_weights(self.policy.net.get_weights())
            self.sampler.send('policy_params', self.policy.net.get_weights())
            self.sampler.send('value_fn_params', self.value_fn.get_weights())

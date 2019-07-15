import copy

import torch
from torch.utils.data import TensorDataset

from gtml.callbacks import CallbackManager
from gtml.constants import DEFAULT_BATCH_SIZE, DEFAULT_GAE_LAMBDA
from gtml.rl.sampling import ParallelSampler
from gtml.train import LossFunction, EpochalMinimizer


class ProximalPolicyOptimization(CallbackManager):
    def __init__(self, env, policy, value_fn, optimizer_factory,
                 clip_eps=0.2, T=128, n_actors=10,
                 c_vf=1.0, c_ent=0.01,
                 batch_size=DEFAULT_BATCH_SIZE, n_epochs=10,
                 gae_lambda=DEFAULT_GAE_LAMBDA):
        self.policy = policy
        self.value_fn = value_fn
        self.old_policy = copy.deepcopy(policy)
        self.clip_eps = clip_eps
        self.T = T
        self.c_vf = c_vf
        self.c_ent = c_ent
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        policy.net.share_memory()
        value_fn.share_memory()
        sampler = ParallelSampler(env, n_actors, {'gae_lambda':gae_lambda})
        sampler.start()
        sampler.send('policy', policy)
        sampler.send('value_fn', value_fn)
        self.sampler = sampler

        L = LossFunction()
        L.add_term('ppo_loss', self.compute_loss)
        all_params = list(policy.parameters()) + list(value_fn.parameters())
        optimizer = optimizer_factory(all_params)
        self.train = EpochalMinimizer(L, optimizer)

        CallbackManager.__init__(self)

    def compute_loss(self, inputs):
        observations, actions, advantages, value_targets = inputs
        action_distributions = self.policy.action_distributions(observations)
        action_distributions_old = self.old_policy.action_distributions(observations)
        r = torch.exp(action_distributions.log_prob(actions) - action_distributions_old.log_prob(actions))
        r_clamped = torch.clamp(r, 1 - self.clip_eps, 1 + self.clip_eps)
        L_clip = torch.min(r * advantages, r_clamped * advantages)
        L_vf = torch.sum((self.value_fn(observations) - value_targets)**2)
        L_ent = action_distributions.entropy()
        L = L_clip - self.c_vf * L_vf + self.c_ent * L_ent
        return -torch.mean(L)

    def run(self, n_iterations=1):
        for itr in range(n_iterations):
            samples = self.sampler.sample(max_steps=self.T)
            dataset = TensorDataset(samples['observations'], samples['actions'],
                                    samples['advantages'], samples['value_targets'])
            self.train.create_data_loader(dataset)
            print('Training...')
            self.train.run(self.n_epochs)
            self.old_policy.net.load_state_dict(self.policy.net.state_dict())
            self.run_callbacks('post-iteration')

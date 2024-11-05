from typing import List

import torch
import torch.nn.functional as F

from force.alg.agent import Agent
from force.data import TensorDataset, TransitionBuffer
from force.defaults import NUMERICAL_EPSILON
from force.env.util import space_shape
from force.nn.models.value_functions import ValueFunction
from force.nn.normalization import Normalizer, InputNormalizerWrapper, InputNormalizedPolicy
from force.nn.optim import Optimizer
from force.nn.util import batch_iterator
from force.policies import GaussianPolicy


def gae(rewards, values, terminals, gamma, lam):
    T = len(rewards)
    assert len(values) == T + 1 # should include value of terminal state
    advs = torch.zeros_like(rewards)
    nonterminals = ~terminals
    lastgaelam = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * nonterminals[t] * values[t+1] - values[t]
        advs[t] = lastgaelam = delta + gamma * lam * nonterminals[t] * lastgaelam
    return advs


class PPO(Agent):
    class Config(Agent.Config):
        policy = GaussianPolicy.Config
        value_function = ValueFunction.Config
        optimizer = Optimizer.Config
        clip_epsilon = 0.2
        epochs_per_batch = 10
        minibatch_size = 64
        normalize_advantages = True
        gae_lambda = 0.95
        max_grad_norm = 0.5
        entropy_bonus = 0.0

    def __init__(self, cfg, obs_space, act_space, device=None):
        super().__init__(cfg, obs_space, act_space, device=device)

        obs_shape = space_shape(obs_space)
        act_shape = space_shape(act_space)
        self.normalizer = Normalizer(obs_shape)
        self.policy = InputNormalizedPolicy(
            GaussianPolicy(cfg.policy, obs_shape, act_shape), self.normalizer
        )
        self.value_function = InputNormalizerWrapper(
            ValueFunction(cfg.value_function, obs_shape), self.normalizer
        )
        self.ac_params = [*self.policy.parameters(), *self.value_function.parameters()]
        self.optimizer = Optimizer(cfg.optimizer, self.ac_params)

    def act(self, observations, eval: bool):
        return self.policy.act(observations, eval)

    def _process_buffers(self, buffers: List[TransitionBuffer]) -> TensorDataset:
        cfg = self.cfg
        obs_list, acts_list, returns_list, values_list, advs_list, gae_returns_list = [], [], [], [], [], []
        for buf in buffers:
            obs, acts, rewards, terminals,  next_obs = buf.get(
                'observations', 'actions', 'rewards', 'terminals', 'next_observations'
            )
            all_obs = torch.cat([obs, next_obs[-1].unsqueeze(dim=0)], dim=0)
            with torch.no_grad():
                values = self.value_function(all_obs)
            advs = gae(rewards, values, terminals,
                       gamma=cfg.discount, lam=cfg.gae_lambda)
            gae_returns = values[:-1] + advs
            obs_list.append(obs)
            acts_list.append(acts)
            advs_list.append(advs)
            gae_returns_list.append(gae_returns)

        return TensorDataset({
            'observations': torch.cat(obs_list, dim=0),
            'actions': torch.cat(acts_list, dim=0),
            'advantages': torch.cat(advs_list, dim=0),
            'gae_returns': torch.cat(gae_returns_list, dim=0)
        })

    def update(self, buffers: list, counters: dict):
        cfg = self.cfg

        # Process (compute returns and stack)
        batch = self._process_buffers(buffers)
        observations, actions, advantages, gae_returns = batch.get()

        # Get log probs under pre-update parameters
        with torch.no_grad():
            old_distr = self.policy.distribution(observations)
            old_log_probs = old_distr.log_prob(actions)
        batch_components = [observations, actions, advantages, gae_returns, old_log_probs]

        # Update observation statistics
        self.normalizer.update(observations)

        for _ in range(cfg.epochs_per_batch):
            for obs, acts, advs, gae_rets, old_lps in batch_iterator(
                batch_components, cfg.minibatch_size, shuffle=True
            ):
                if cfg.normalize_advantages:
                    advs = (advs - advs.mean()) / (advs.std() + NUMERICAL_EPSILON)

                # Policy objective (to be maximized; will be negated below)
                policy_distr = self.policy.distribution(obs)
                log_probs = policy_distr.log_prob(acts)
                ratios = torch.exp(log_probs - old_lps)
                clipped_ratios = ratios.clamp(
                    min = 1 - cfg.clip_epsilon,
                    max = 1 + cfg.clip_epsilon
                )
                policy_objective = torch.minimum(ratios * advs, clipped_ratios * advs)

                if cfg.entropy_bonus > 0:
                    try:
                        entropy = policy_distr.entropy()
                    except NotImplementedError:
                        # Distribution does not implement entropy().
                        # Use sample-based approximation instead.
                        entropy = -policy_distr.log_prob(policy_distr.sample().detach())
                    mean_entropy = entropy.mean()
                    policy_objective = policy_objective + cfg.entropy_bonus * mean_entropy

                # Value loss
                values = self.value_function(obs)
                value_loss = F.mse_loss(values, gae_rets)

                loss = -policy_objective.mean() + value_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac_params, cfg.max_grad_norm)
                self.optimizer.step()
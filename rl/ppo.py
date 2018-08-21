import copy
import torch

from gtml.defaults import BATCHSIZE, GAE_LAMBDA
from gtml.nn.opt import Minimizer
from gtml.rl.util import estimate_advantages_and_value_targets


class ProximalPolicyOptimization(Minimizer):
    def __init__(self, ac, optimizer=None, batchsize=BATCHSIZE, epsilon=0.2,
            N=10, T=20, K=10, c_vf=1.0, c_ent=0.01, gae_lambda=GAE_LAMBDA):
        self.ac = ac
        self.old_policy = copy.deepcopy(ac.policy)
        self.epsilon = epsilon
        self.N = N
        self.T = T
        self.K = K
        self.c_vf = c_vf
        self.c_ent = c_ent
        self.gae_lambda = gae_lambda

        def compute_loss(observations, actions, advantages, value_targets):
            action_distributions = self.ac.policy.action_distributions(observations)
            action_distributions_old = self.old_policy.action_distributions(observations)
            r = torch.exp(action_distributions.log_prob(actions) - action_distributions_old.log_prob(actions))
            L_clip = torch.min(r * advantages, torch.clamp(r, 1 - epsilon, 1 + epsilon) * advantages)
            L_vf = torch.sum((self.ac.value_fn(observations) - value_targets)**2)
            # L_ent = policy_dist.entropy()
            L = L_clip - self.c_vf * L_vf # + self.c_ent * L_ent
            return -torch.mean(L)

        Minimizer.__init__(self, ac.parameters(), compute_loss,
                optimizer=optimizer, batchsize=batchsize)

    def run(self, engine, n_iterations=1):
        episode = engine.new_episode()
        for itr in range(n_iterations):
            observations = []
            actions = []
            advantages = []
            value_targets = []
            for _ in range(self.N):
                steps_taken = episode.run(self.ac.policy, self.T)
                new_observations = episode.observations[-(steps_taken+1):]
                new_values = torch.squeeze(self.ac.value_fn(new_observations).data)
                new_actions = episode.actions[-steps_taken:]
                new_rewards = episode.rewards[-steps_taken:]
                new_advantages, new_value_targets = estimate_advantages_and_value_targets(
                        new_rewards, episode.done, new_values,
                        gamma=engine.env.discount, lam=self.gae_lambda
                )
                observations.append(new_observations[:-1])
                actions.append(new_actions)
                advantages.append(new_advantages)
                value_targets.append(new_value_targets)
                if episode.done:
                    episode = engine.new_episode()

            update_inputs = (
                torch.cat(observations),
                torch.cat(actions),
                torch.cat(advantages),
                torch.cat(value_targets)
            )
            for _ in range(self.K):
                self.epoch(update_inputs)
            self.old_policy.net.load_state_dict(self.ac.policy.net.state_dict())

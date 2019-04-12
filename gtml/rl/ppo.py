import copy
import torch

from gtml.constants import DEFAULT_BATCHSIZE, DEFAULT_GAE_LAMBDA
from gtml.nn.opt import Minimizer
from gtml.rl.core import Episode
from gtml.rl.env import *
from gtml.rl.gae import estimate_advantages_and_value_targets


class ProximalPolicyOptimization(Minimizer):
    def __init__(self, ac, optimizer=None, batchsize=DEFAULT_BATCHSIZE,
                 epsilon=0.2, N=10, T=20, K=10, c_vf=1.0, c_ent=0.01,
                 gae_lambda=DEFAULT_GAE_LAMBDA):
        self.ac = ac
        self.old_policy = copy.deepcopy(ac.policy)
        self.epsilon = epsilon
        self.N = N
        self.T = T
        self.K = K
        self.c_vf = c_vf
        self.c_ent = c_ent
        self.gae_lambda = gae_lambda
        self.total_steps = 0
        self.episode_rewards = []

        def compute_loss(observations, actions, advantages, value_targets):
            action_distributions = self.ac.policy.action_distributions(observations)
            action_distributions_old = self.old_policy.action_distributions(observations)
            r = torch.exp(action_distributions.log_prob(actions) - action_distributions_old.log_prob(actions))
            L_clip = torch.min(r * advantages, torch.clamp(r, 1 - epsilon, 1 + epsilon) * advantages)
            L_vf = torch.sum((self.ac.value_fn(observations) - value_targets)**2)
            L_ent = action_distributions.entropy()
            L = L_clip - self.c_vf * L_vf + self.c_ent * L_ent
            return -torch.mean(L)

        Minimizer.__init__(self, ac.parameters(), compute_loss,
                optimizer=optimizer, batchsize=batchsize)

    def run(self, env, n_iterations=1):
        episode = Episode(env)
        for itr in range(n_iterations):
            observations, actions, advantages, value_targets = [], [], [], []
            for _ in range(self.N):
                steps_taken = episode.run(self.ac.policy, self.T)
                self.total_steps += steps_taken
                new_observations = episode.observations[-(steps_taken+1):]
                new_values = torch.squeeze(self.ac.value_fn(new_observations).data)
                new_actions = episode.actions[-steps_taken:]
                new_rewards = episode.rewards[-steps_taken:]
                new_advantages, new_value_targets = estimate_advantages_and_value_targets(
                        new_rewards, episode.done, new_values,
                        gamma=env.discount, lam=self.gae_lambda
                )
                observations.append(new_observations[:-1])
                actions.append(new_actions)
                advantages.append(new_advantages)
                value_targets.append(new_value_targets)
                if episode.done:
                    self.episode_rewards.append(rewards)
                    self.run_callbacks('post-episode')
                    episode = Episode(env)

            for _ in range(self.K):
                self.epoch(torch.cat(observations),
                           torch.cat(actions),
                           torch.cat(advantages),
                           torch.cat(value_targets))

            self.old_policy.net.load_state_dict(self.ac.policy.net.state_dict())
            self.run_callbacks('post-iteration')

    def run_parallel(self, envname, n_iterations=1):
        import multiprocessing as mp
        data_q = mp.Queue()
        workers = []
        for _ in range(self.N):
            ac_q = mp.Queue()
            proc = mp.Process(target=_ppo_worker, args=(ac_q, data_q, envname, self.ac, self.T, self.gae_lambda))
            proc.start()
            workers.append((ac_q, proc))

        for itr in range(n_iterations):
            observations, actions, advantages, value_targets = [], [], [], []
            for (ac_q, proc) in workers:
                ac_q.put(self.ac.state_dict())

            for i in range(self.N):
                new_observations, new_actions, new_advantages, new_value_targets, total_reward = data_q.get()
                observations.append(new_observations)
                actions.append(new_actions)
                advantages.append(new_advantages)
                value_targets.append(new_value_targets)
                if total_reward is not None:
                    self.episode_rewards.append(total_reward)
                    self.run_callbacks('post-episode')

            for _ in range(self.K):
                self.epoch(torch.cat(observations),
                           torch.cat(actions),
                           torch.cat(advantages),
                           torch.cat(value_targets))

            self.old_policy.net.load_state_dict(self.ac.policy.net.state_dict())
            self.run_callbacks('post-iteration')

        for (proc, conn) in workers:
            proc.join()

def _ppo_worker(ac_q, data_q, envname, ac, T, gae_lambda):
    env = AtariEnvironment(envname) if envname.split('-')[0] in ATARI_NAMES else Environment(envname)
    episode = Episode(env)
    while True:
        ac.load_state_dict(ac_q.get())
        steps_taken = episode.run(ac.policy, T)
        observations = episode.observations[-(steps_taken+1):]
        values = torch.squeeze(ac.value_fn(observations))
        actions = episode.actions[-steps_taken:]
        rewards = episode.rewards[-steps_taken:]
        advantages, value_targets = estimate_advantages_and_value_targets(
                rewards, episode.done, values,
                gamma=env.discount, lam=gae_lambda
        )
        data_q.put((observations[:-1], actions, advantages, value_targets, episode.total_reward if episode.done else None))
        if episode.done:
            episode = Episode(env)

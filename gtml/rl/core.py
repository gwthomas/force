from collections import defaultdict
import gym
import os
import torch


class Episode:
    def __init__(self, env):
        self.env = env
        self.t = 0
        self.lobservations = []
        self.lactions = []
        self.lrewards = []
        self.done = False

    def latest_observation(self):
        return self.lobservations[-1] if self.lobservations else self.observations[-1]

    # Act for a given number of steps or until the episode ends
    def run(self, policy, max_steps):
        taken = 0

        if self.t == 0:
            observation = self.env.reset()
            self.lobservations.append(observation)

        while taken < max_steps:
            self.env.maybe_render()
            action = policy.act1(self.latest_observation())
            if self.env.discrete_actions:
                action = int(action)
            self.lactions.append(action)
            observation, reward, done, info = self.env.step(action)
            self.lobservations.append(observation)
            self.lrewards.append(reward)
            self.done = done
            taken += 1
            self.t += 1
            if done:
                break

        self._commit()
        return taken

    def _commit(self):
        if self.lobservations is None:
            return

        self.observations = torch.stack(self.lobservations)
        self.rewards = torch.Tensor(self.lrewards)
        if self.env.discrete_actions:
            self.actions = torch.Tensor(self.lactions)
        else:
            self.actions = torch.stack(self.lactions)

        if self.done:
            self.total_reward = self.rewards.sum().item()
            self.lobservations = None
            self.lactions = None
            self.lrewards = None

def rollout(env, policy, horizon=None):
    if horizon is None:
        horizon = float('inf')

    episode = Episode(env)
    episode.run(policy, horizon)
    return episode

def rollouts(env, policy, num_episodes, horizon=None):
    return [rollout(env, policy, horizon=horizon) for _ in range(num_episodes)]

def evaluate(env, policy, num_episodes=1, horizon=None):
    episodes = rollouts(env, policy, num_episodes, horizon=horizon)
    return torch.tensor([episode.total_reward for episode in episodes])

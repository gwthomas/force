from collections import defaultdict
import gym
import os
import torch

from gtml.common.callbacks import CallbackManager
from gtml.common.memory import Memory


class Episode:
    def __init__(self, engine):
        self.engine = engine
        self.t = 0
        self.lobservations = []
        self.lactions = []
        self.lrewards = []
        self.done = False

    def __len__(self):
        return len(self.actions)

    def latest_observation(self):
        return self.lobservations[-1] if len(self.lobservations) > 0 else None

    def latest_action(self):
        return self.lactions[-1] if len(self.lactions) > 0 else None

    def latest_reward(self):
        return self.lrewards[-1] if len(self.lrewards) > 0 else None

    def latest_transition(self):
        if len(self.lobservations) < 2:
            return None
        observation, next_observation = self.lobservations[-2:]
        action = self.lactions[-1]
        reward = self.lrewards[-1]
        return observation, action, next_observation, reward

    def commit(self):
        if self.lobservations is None:
            return

        self.observations = torch.stack(self.lobservations)
        self.rewards = torch.Tensor(self.lrewards)
        if self.engine.env.discrete_actions:
            self.actions = torch.Tensor(self.lactions)
        else:
            self.actions = torch.stack(self.lactions)

        # if self.done:
        #     self.lobservations = None
        #     self.lactions = None
        #     self.lrewards = None

    # Act for a given number of steps or until the episode ends
    def run(self, policy, steps):
        env = self.engine.env
        taken = 0
        if self.t == 0:
            observation = env.reset()
            observation = torch.Tensor(observation)
            self.lobservations.append(observation)

        while taken < steps:
            if self.engine.render:
                env.render()
            action = policy.act1(self.latest_observation())
            if env.discrete_actions:
                action = int(action)
            self.lactions.append(action)
            next_observation, reward, done, info = env.step(action)
            next_observation = torch.Tensor(next_observation)
            self.lobservations.append(next_observation)
            self.lrewards.append(reward)
            self.done = done
            taken += 1
            self.t += 1
            self.engine.global_step += 1
            self.engine.run_callbacks('post-step', episode=self)
            if done:
                break

        self.commit()
        if self.done:
            self.engine.run_callbacks('post-episode', episode=self)

        return taken


class Engine(CallbackManager):
    def __init__(self, env, episode_memory=1000, render=False):
        CallbackManager.__init__(self)
        self.env = env
        self.global_step = 0
        self.episodes = Memory(episode_memory)
        self.render = render

    def new_episode(self):
        episode = Episode(self)
        self.episodes.add(episode)
        return episode

    def rollout(self, policy, horizon=None):
        if horizon is None:
            horizon = float('inf')

        episode = self.new_episode()
        episode.run(policy, horizon)
        return episode

    def rollouts(self, policy, num_episodes, horizon=None):
        return [self.rollout(policy, horizon=horizon) for _ in range(num_episodes)]

    def evaluate(self, policy, num_episodes=1, horizon=None):
        episodes = self.rollouts(policy, num_episodes, horizon=horizon)
        return torch.tensor([episode.discounted_return for episode in episodes])

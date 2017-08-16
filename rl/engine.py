from collections import defaultdict
import numpy as np
import os
from scipy.signal import lfilter

from gtml.common.callbacks import CallbackManager
from gtml.common.memory import Memory
from gtml.common.tf import get_sess


class Episode:
    def __init__(self, engine):
        self.engine = engine
        self.t = 0
        self.raw_observations = []
        self.observations = []
        self.actions = []
        self.rewards = []
        self.done = False

    def latest_observation(self):
        return self.observations[-1] if len(self.observations) > 0 else None

    def latest_action(self):
        return self.actions[-1] if len(self.actions) > 0 else None

    def latest_reward(self):
        return self.rewards[-1] if len(self.rewards) > 0 else None

    def latest_transition(self):
        observation, next_observation = self.observations[-2:]
        action = self.actions[-1]
        reward = self.rewards[-1]
        return observation, action, next_observation, reward

    def finalize(self):
        for key in ('raw_observations', 'observations', 'actions', 'rewards'):
            setattr(self, key, np.array(getattr(self, key)))

    def __len__(self):
        return len(self.actions)

    # Act for a given number of steps or until the episode ends
    def run(self, policy, steps):
        sess = get_sess(self.engine.sess)
        env = self.engine.env
        taken = 0
        if self.t == 0:
            observation = env.reset()
            self.observations.append(observation)

        while taken < steps:
            if self.engine.render:
                env.render()
            action = policy.act([self.latest_observation()], sess=sess)[0]
            self.actions.append(action)
            next_observation, reward, done, info = env.step(action)
            self.observations.append(next_observation)
            self.rewards.append(reward)
            self.done = done
            taken += 1
            self.t += 1
            self.engine.global_step += 1
            self.engine.run_callbacks('post-step', episode=self)
            if done:
                break

        if self.done:
            self.finalize()
            self.total_reward = np.sum(self.rewards)
            self.engine.run_callbacks('post-episode', episode=self)

        return taken


class Engine(CallbackManager):
    def __init__(self, env, episode_memory=1, render=False, sess=None):
        CallbackManager.__init__(self)
        self.env = env
        self.sess = sess
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
        return np.array([episode.discounted_return for episode in episodes])

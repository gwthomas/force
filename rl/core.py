from collections import defaultdict
import numpy as np
import os

from gtml.util.memory import Memory
from gtml.util.tf import get_sess


def discounted_sum(rewards, discount):
    return np.sum(np.array(rewards) * discount**np.arange(len(rewards)))

def discounted_returns(rewards, discount):
    n = len(rewards)
    R = rewards[-1]
    returns = np.zeros_like(rewards)
    for i in range(n-1, -1, -1):
        R = rewards[i] + discount * R
        returns[i] = R
    return returns


class Episode:
    def __init__(self, engine):
        self.engine = engine
        self.t = 0
        self.raw_observations = []
        self.observations = []
        self.actions = []
        self.rewards = []
        self.policy_outputs = defaultdict(list)
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
        env = policy.env
        taken = 0
        if self.t == 0:
            observation = env.reset()
            self.observations.append(observation)

        while taken < steps:
            if self.engine.render:
                env.render()
            results = policy.act([self.latest_observation()], sess=sess)
            for key, value in results.items():
                self.policy_outputs[key].extend(value)
            action = results['_actions'][0]
            self.actions.append(action)
            next_observation, reward, done, info = env.step(action)
            self.observations.append(next_observation)
            self.rewards.append(reward)
            self.done = done
            taken += 1
            self.t += 1
            self.engine.global_step += 1
            self.engine.run_callbacks('post-step')
            if done:
                break

        if self.done:
            self.observations.pop()
            self.finalize()
            self.discounted_return = discounted_sum(self.rewards, env.discount)
            self.engine.run_callbacks('post-episode')

        return taken


class Engine:
    def __init__(self, log_dir=None, episode_memory=10, render=False, sess=None):
        self.set_log_dir(log_dir)
        self.global_step = 0
        self.render = render
        self.sess = sess
        self.episodes = Memory(episode_memory)
        self._callbacks = defaultdict(list)

    def log_path(self, filename):
        return os.path.join(self.log_dir, filename)

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir
        if log_dir is not None and not os.path.isdir(log_dir):
            print('Creating log directory at', log_dir)
            os.makedirs(log_dir)

    def add_callback(self, event, callback):
        self._callbacks[event].append(callback)

    def run_callbacks(self, event):
        for callback in self._callbacks[event]:
            callback(self)

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

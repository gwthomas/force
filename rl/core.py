from collections import defaultdict
import numpy as np

from gtml.util.tf import get_sess

class Episode:
    def __init__(self):
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
    def run(self, policy, steps, sess=None, render=False):
        sess = get_sess(sess)
        env = policy.env
        taken = 0
        if self.t == 0:
            observation = env.reset()
            self.observations.append(observation)

        while taken < steps:
            if render:
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
            if done:
                break

        if self.done:
            self.observations.pop()
            self.finalize()
            self.discounted_return = discounted_sum(self.rewards, env.discount)

        return taken


def discounted_sum(rewards, discount):
    return np.sum(np.array(rewards) * discount**np.arange(len(rewards)))


def rollout(policy, horizon=None, render=False):
    if horizon is None:
        horizon = float('inf')

    episode = Episode()
    episode.run(policy, horizon, render=render)
    return episode

def rollouts(policy, num_episodes, horizon=None, render=False):
    return [rollout(policy, horizon=horizon, render=render) for _ in range(num_episodes)]

def evaluate(policy, num_episodes=1, horizon=None, render=False):
    episodes = rollouts(policy, num_episodes, horizon=horizon, render=render)
    return np.array([episode.discounted_return for episode in episodes])

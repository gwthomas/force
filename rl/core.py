from multiprocessing.pool import Pool

from gtml.util.memory import Memory
import numpy as np


class Episode:
    def __init__(self):
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


def discounted_sum(rewards, discount):
    return np.sum(np.array(rewards) * discount**np.arange(len(rewards)))

def discounted_returns(rewards, discount):
    # There is almost certainly a more efficient way to implement this
    # but it's good enough for now
    return np.array([np.sum(rewards[t:]) for t in range(len(rewards))])


# Act for a given number of steps or until the episode ends
def partial_rollout(policy, episode, steps, render=False):
    env = policy.env
    taken = 0
    if episode.t == 0:
        observation = env.reset()
        episode.observations.append(observation)

    while taken < steps:
        if render:
            env.render()
        action = policy.get_action(episode.latest_observation())
        episode.actions.append(action)
        next_observation, reward, done, info = env.step(action)
        episode.observations.append(next_observation)
        episode.rewards.append(reward)
        episode.done = done
        taken += 1
        episode.t += 1
        if done:
            break

    if episode.done:
        episode.observations.pop()
        episode.finalize()
        episode.discounted_return = discounted_sum(episode.rewards, env.discount)

    return taken


def rollout(policy, horizon=None, render=False):
    if horizon is None:
        horizon = float('inf')

    episode = Episode()
    partial_rollout(policy, episode, horizon, render=render)
    return episode

def rollouts(policy, num_episodes, horizon=None, render=False, num_processes=1):
    if num_processes == 1:
        return [rollout(policy, horizon=horizon, render=render) for _ in range(num_episodes)]
    else:
        pool = Pool(num_processes)

def evaluate(policy, num_episodes=1, horizon=None, render=False, num_processes=1):
    episodes = rollouts(policy, num_episodes, horizon=horizon, render=render, num_processes=num_processes)
    return np.array([episode.discounted_return for episode in episodes])

from gtml.util.memory import Memory
from gtml.util.rl import discounted_sum
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


# Provides a common interface for both training and testing policies.
# Use callbacks to update parameters, save progress to disk, etc.
class Engine(object):
    EVENTS = ('pre-step', 'post-step', 'pre-episode', 'post-episode')

    def __init__(self,
            discount=1,
            episode_memory_size=1 # How many episodes to keep around at a time
    ):
        self.discount = discount
        self.episode_memory_size = episode_memory_size
        self._callbacks = {event:{} for event in Engine.EVENTS}

    def reset(self):
        self.itr = 0
        self.episodes = Memory(self.episode_memory_size)

    def new_episode(self):
        episode = Episode()
        self.episodes.add(episode)
        return episode

    @property
    def num_episodes(self):
        return len(self.episodes)

    def latest_episode(self):
        return self.episodes[-1] if len(self.episodes) > 0 else None

    def recent_episodes(self, n):
        return self.episodes[-n:] if len(self.episodes) >= n else None

    def register_callback(self, event, name, callback):
        assert event in Engine.EVENTS
        self._callbacks[event][name] = callback

    def unregister_callback(self, name):
        for event in Engine.EVENTS:
            callbacks = self._callbacks[event]
            if name in callbacks:
                del callbacks[name]

    def _run_callbacks(self, event, **kwargs):
        episode = self.latest_episode()
        for name, callback in self._callbacks[event].items():
            callback(self, episode)

    def run(self, policy,
            phi=lambda obs, unused: obs,   # Preprocessor for observations
            num_episodes=1,
            T=1000,
            render=False):
        self.reset()
        env = policy.env
        discounted_returns = []
        for episode_num in range(num_episodes):
            self._run_callbacks('pre-episode')
            episode = self.new_episode()
            raw_observation = env.reset()
            episode.raw_observations.append(raw_observation)
            observation = phi(raw_observation, self)
            episode.observations.append(observation)
            for t in range(T):
                if render:
                    env.render()
                episode.t = t
                self._run_callbacks('pre-step')
                action = policy.get_action(episode.latest_observation())
                episode.actions.append(action)
                raw_next_observation, reward, done, info = env.step(action)
                episode.raw_observations.append(raw_next_observation)
                episode.rewards.append(reward)
                episode.done = done
                next_observation = phi(raw_next_observation, self)
                episode.observations.append(next_observation)
                self._run_callbacks('post-step')
                if episode.done:
                    break
                self.itr += 1
            episode.finalize()
            episode.discounted_return = discounted_sum(episode.rewards, self.discount)
            discounted_returns.append(episode.discounted_return)
            self._run_callbacks('post-episode')
        return discounted_returns

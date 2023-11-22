import gymnasium as gym
import numpy as np
import torch

from force.data import CircularDataset
from force.env.util import space_dim, get_max_episode_steps
from force.env.batch import BatchedEnv
from force.nn.util import torchify, numpyify, random_indices
from force.policies import BasePolicy
from force.util import discounted_sum


class ReplayBuffer(CircularDataset):
    def __init__(self, observation_space, action_space, capacity,
                 device=None):
        self._observation_space = observation_space
        self._action_space = action_space

        obs_dim = space_dim(observation_space)
        discrete_actions = isinstance(action_space, gym.spaces.Discrete)
        if discrete_actions:
            action_dtype = torch.int
            action_shape = ()
        else:
            action_dtype = torch.float
            action_shape = (space_dim(action_space),)

        components = {
            'observations': (torch.float, (obs_dim,)),
            'actions': (action_dtype, action_shape),
            'next_observations': (torch.float, (obs_dim,)),
            'rewards': (torch.float, ()),
            'terminals': (torch.bool, ()),
            'truncateds': (torch.bool, ())
        }
        super().__init__(components, capacity, device)



class SimpleSampler:
    """
    For sampling individual steps/transitions.
    """
    def __init__(self, env):
        self._env = env
        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space
        self._steps_done = 0
        self._episode_returns = []
        self.max_episode_steps = get_max_episode_steps(self._env)
        self.reset()

    @property
    def steps_done(self):
        return self._steps_done

    def get_returns(self, clear=True):
        returns = list(self._episode_returns)
        if clear:
            self._episode_returns.clear()
        return returns

    def reset(self):
        self._last_obs, _ = self._env.reset()
        self._t = 0
        self._episode_return = 0.

    def run(self, policy: BasePolicy, num_steps: int, eval: bool = False, buffer: ReplayBuffer =None):
        if buffer is None:
            # Create buffer to store samples
            buffer = ReplayBuffer(self._observation_space, self._action_space, num_steps)

        for _ in range(num_steps):
            action = numpyify(policy.act1(self._last_obs, eval))
            assert self._env.action_space.contains(action)
            next_obs, reward, terminated, truncated, info = self._env.step(action)
            self._steps_done += 1
            self._t += 1
            self._episode_return += reward
            buffer.append({
                'observations': self._last_obs,
                'actions': action,
                'next_observations': next_obs,
                'rewards': reward,
                'terminals': terminated,
                'truncateds': truncated
            })
            if terminated or truncated:
                self._episode_returns.append(self._episode_return)
                self.reset()
            else:
                self._last_obs = next_obs
        return buffer


def sample_episode(env, policy, eval=False,
                   max_steps=None,
                   recorder=None, render=False):
    T = max_steps if max_steps is not None else get_max_episode_steps(env)
    episode = ReplayBuffer(env.observation_space, env.action_space, T)
    obs, info = env.reset()

    if recorder:
        recorder.capture_frame()
    elif render:
        env.unwrapped.render()

    for t in range(T):
        action = policy.act1(obs, eval=eval)
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode.append({
            'observations': obs,
            'actions': action,
            'next_observations': next_obs,
            'rewards': reward,
            'terminals': terminated,
            'truncateds': truncated
        })

        if recorder:
            recorder.capture_frame()
        elif render:
            env.unwrapped.render()

        if terminated or truncated:
            break
        else:
            obs = next_obs.clone()

    return episode


def sample_episodes_batched(env, policy, n_traj, eval=False, max_steps=None):
    assert isinstance(env, BatchedEnv)

    T = max_steps if max_steps is not None else get_max_episode_steps(env)
    traj_buffer_factory = lambda: ReplayBuffer(env.observation_space, env.action_space, T)
    traj_buffers = [traj_buffer_factory() for _ in range(env.num_envs)]
    complete_episodes = []

    obs = env.reset()
    while True:
        with torch.no_grad():
            actions = policy.act(obs, eval=eval)
        next_obs, rewards, terminals, truncateds, infos = env.step(actions)

        _next_obs = next_obs.clone()
        reset_indices = []

        for i in range(env.num_envs):
            traj_buffers[i].append({
                'observations': obs[i],
                'actions': actions[i],
                'next_observations': next_obs[i],
                'rewards': rewards[i],
                'terminals': terminals[i],
                'truncateds': truncateds[i]
            })
            if terminals[i] or len(traj_buffers[i]) == T:
                complete_episodes.append(traj_buffers[i])
                if len(complete_episodes) == n_traj:
                    # Done!
                    return complete_episodes

                reset_indices.append(i)
                traj_buffers[i] = traj_buffer_factory()

        if reset_indices:
            reset_indices = np.array(reset_indices)
            _next_obs[reset_indices] = env.partial_reset(reset_indices)

        obs = _next_obs.clone()


def evaluate_policy(env, policy, n_episodes=10, discount=1):
    if not isinstance(env, BatchedEnv):
        env = BatchedEnv([env])

    returns = []
    for episode in sample_episodes_batched(env, policy, n_episodes, eval=True):
        states, actions, next_states, rewards, terminals = episode.get()
        returns.append(discounted_sum(rewards, discount))
    return torchify(returns)
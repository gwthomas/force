import numpy as np
import torch

from force.env.util import env_dims, isdiscrete
from force.env.batch import BaseBatchedEnv, ProductEnv
from force.torch_util import device, Module, torchify, random_indices
from force.util import discounted_sum


class SampleBuffer(Module):
    COMPONENT_NAMES = ('states', 'actions', 'next_states', 'rewards', 'dones')

    def __init__(self, state_dim, action_dim, capacity, discrete_actions=False):
        super().__init__()
        if discrete_actions:
            assert action_dim == 1
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.discrete_actions = discrete_actions
        self.register_buffer('_states', torch.FloatTensor(capacity, state_dim))
        self.register_buffer('_actions', torch.IntTensor(capacity) if discrete_actions else \
                                         torch.FloatTensor(capacity, action_dim))
        self.register_buffer('_next_states', torch.FloatTensor(capacity, state_dim))
        self.register_buffer('_rewards', torch.FloatTensor(capacity))
        self.register_buffer('_dones', torch.BoolTensor(capacity))
        self.register_buffer('_pointer', torch.tensor(0, dtype=torch.long))

    def __len__(self):
        return min(self._pointer, self.capacity)

    def _get1(self, name):
        assert name in SampleBuffer.COMPONENT_NAMES
        buf = getattr(self, f'_{name}')
        if self._pointer <= self.capacity:
            return buf[:self._pointer]
        else:
            i = self._pointer % self.capacity
            return torch.cat([buf[i:], buf[:i]])

    def get(self, *names, to_device=True):
        """
        Retrieves data from the buffer. Pass a vararg list of names.
        What is returned depends on how many names are given:
            * a list of all (5) components if no names are given
            * a single component if one name is given
            * a list with one component for each name otherwise
        """
        if len(names) == 0:
            names = SampleBuffer.COMPONENT_NAMES
        bufs = [self._get1(name) for name in names]
        if to_device:
            bufs = [buf.to(device) for buf in bufs]
        return bufs if len(bufs) > 1 else bufs[0]

    def append(self, state, action, next_state, reward, done):
        i = self._pointer % self.capacity
        self._states[i] = state
        self._actions[i] = action
        self._next_states[i] = next_state
        self._rewards[i] = reward
        self._dones[i] = done
        self._pointer += 1

    def extend(self, states, actions, next_states, rewards, dones):
        batch_size = len(states)
        assert batch_size <= self.capacity, 'We do not support extending by more than buffer capacity'
        i = self._pointer % self.capacity
        end = i + batch_size
        if end <= self.capacity:
            self._states[i:end] = states
            self._actions[i:end] = actions
            self._next_states[i:end] = next_states
            self._rewards[i:end] = rewards
            self._dones[i:end] = dones
        else:
            fit = self.capacity - i
            overflow = end - self.capacity
            # Note: fit + overflow = batch_size
            self._states[-fit:] = states[:fit]
            self._actions[-fit:] = actions[:fit]
            self._next_states[-fit:] = next_states[:fit]
            self._rewards[-fit:] = rewards[:fit]
            self._dones[-fit:] = dones[:fit]
            self._states[:overflow] = states[-overflow:]
            self._actions[:overflow] = actions[-overflow:]
            self._next_states[:overflow] = next_states[-overflow:]
            self._rewards[:overflow] = rewards[-overflow:]
            self._dones[:overflow] = dones[-overflow:]
        self._pointer += batch_size

    def sample(self, batch_size, replace=False, to_device=True):
        indices = random_indices(len(self), size=batch_size, replace=replace)
        bufs = [
            self._states[indices],
            self._actions[indices],
            self._next_states[indices],
            self._rewards[indices],
            self._dones[indices]
        ]
        return [buf.to(device) for buf in bufs] if to_device else bufs


def concat_sample_buffers(buffers):
    state_dim, action_dim = buffers[0].state_dim, buffers[0].action_dim
    discrete_actions = buffers[0].discrete_actions
    total_capacity = 0
    for buffer in buffers:
        assert buffer.state_dim == state_dim
        assert buffer.action_dim == action_dim
        assert buffer.discrete_actions == discrete_actions
        total_capacity += len(buffer)
    combined_buffer = SampleBuffer(state_dim, action_dim, total_capacity,
                                   discrete_actions=discrete_actions)
    for buffer in buffers:
        combined_buffer.extend(*buffer.get())
    return combined_buffer


class StepSampler:
    """
    For sampling individual steps/transitions. Not suitable for trajectories (use sample_trajectories below)
    """
    def __init__(self, env):
        self.env = env if isinstance(env, BaseBatchedEnv) else ProductEnv([env])
        self.samples_taken = 0
        self.reset()

    def reset(self):
        self.set_states(self.env.reset(), set_env_states=False)

    def set_states(self, states, set_env_states=True):
        self._states = states.clone()
        if set_env_states:
            self.env.set_states(states)
        self._n_steps = torch.zeros(self.env.n_envs, device=device)

    def run(self, policy, n_samples=None, n_steps=None, given_buffer=None, eval=False, post_step_callback=None):
        if n_samples is None and n_steps is None:
            raise ValueError('StepSampler.run() must be passed n_samples or n_steps')
        elif n_samples is not None and n_steps is not None:
            raise ValueError('StepSampler.run() cannot be passed both n_samples and n_steps')
        elif n_samples is None:
            assert isinstance(n_steps, int)
            n_samples = n_steps * self.env.n_envs
        elif n_steps is None:
            assert isinstance(n_samples, int)
            assert n_samples % self.env.n_envs == 0, f'n_samples ({n_samples}) is not divisible by n_envs {self.env.n_envs}'
            n_steps = n_samples // self.env.n_envs

        state_dim, action_dim = env_dims(self.env)
        buffer = given_buffer if given_buffer is not None else SampleBuffer(state_dim, action_dim, n_samples)
        for t in range(n_steps):
            actions = policy.act(self._states, eval)
            next_states, rewards, dones, infos = self.env.step(actions)
            buffer.extend(self._states, actions, next_states, rewards, dones)
            self._n_steps += 1
            if callable(post_step_callback):
                post_step_callback(buffer)
            timeouts = self._n_steps == self.env._max_episode_steps
            indices = torch.nonzero(dones | timeouts).flatten()
            if len(indices) > 0:
                next_states = next_states.clone()
                next_states[indices] = self.env.partial_reset(indices)
                self._n_steps[indices] = 0
            self._states.copy_(next_states)
        self.samples_taken += n_samples
        return buffer


def sample_trajectories(env, policy, n_traj, eval=False):
    if not isinstance(env, BaseBatchedEnv):
        env = ProductEnv([env])

    state_dim, action_dim = env_dims(env)
    discrete_actions = isdiscrete(env.action_space)
    traj_buffer_factory = lambda: SampleBuffer(state_dim, 1 if discrete_actions else action_dim, env._max_episode_steps,
                                               discrete_actions=discrete_actions)
    traj_buffers = [traj_buffer_factory() for _ in range(env.n_envs)]
    complete_trajectories = []

    states = env.reset()
    while True:
        actions = policy.act(states, eval=eval)
        next_states, rewards, dones, infos = env.step(actions)

        _next_states = next_states.clone()
        reset_indices = []

        for i in range(env.n_envs):
            traj_buffers[i].append(states[i], actions[i], next_states[i], rewards[i], dones[i])
            if dones[i] or len(traj_buffers[i]) == env._max_episode_steps:
                complete_trajectories.append(traj_buffers[i])

                if len(complete_trajectories) == n_traj:
                    return complete_trajectories

                reset_indices.append(i)
                traj_buffers[i] = traj_buffer_factory()

        if reset_indices:
            reset_indices = np.array(reset_indices)
            _next_states[reset_indices] = env.partial_reset(reset_indices)

        states.copy_(_next_states)


def evaluate_policy(env, policy, n_episodes=10, discount=1, reward_function=None):
    returns = []
    for traj in sample_trajectories(env, policy, n_episodes, eval=True):
        states, actions, next_states, rewards, dones = traj.get()
        if reward_function is not None:
            rewards = reward_function(states, actions, next_states)
        returns.append(discounted_sum(rewards, discount))
    return torchify(returns)
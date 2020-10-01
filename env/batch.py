from abc import ABC, abstractmethod
import gym
import torch

from force.torch_util import device
from .util import get_max_episode_steps


class BaseBatchedEnv(gym.Env, ABC):
    def __init__(self, proto_env, n_envs, max_episode_steps=None):
        self.proto_env = proto_env
        self.n_envs = n_envs
        if max_episode_steps is not None:
            self._max_episode_steps = max_episode_steps
        else:
            self._max_episode_steps = get_max_episode_steps(proto_env)

    @property
    def observation_space(self):
        return self.proto_env.observation_space

    @property
    def action_space(self):
        return self.proto_env.action_space

    @abstractmethod
    def _reset_index(self, index):
        pass

    def partial_reset(self, indices):
        return torch.stack([self._reset_index(index) for index in indices])

    def reset(self):
        return self.partial_reset(torch.arange(self.n_envs))

    @abstractmethod
    def _step(self, actions):
        pass

    def step(self, actions):
        return self._step(actions)

    def get_states(self):
        raise NotImplementedError

    def set_states(self, states, indices):
        raise NotImplementedError


class StatefulBatchedEnv(BaseBatchedEnv):
    _state_readonly = True

    def __init__(self, proto_env, n_envs, max_episode_steps=None):
        super().__init__(proto_env, n_envs, max_episode_steps)
        state_dim = self.observation_space.shape[0]
        self._states = torch.zeros(self.n_envs, state_dim, device=device)

    def partial_reset(self, indices):
        initial_states = super().partial_reset(indices)
        self._states[indices] = initial_states
        return initial_states

    def step(self, actions):
        next_states, rewards, dones, infos = self._step(actions)
        self._states.copy_(next_states)
        return next_states, rewards, dones, infos

    def get_states(self):
        return self._states.clone()

    def set_states(self, states, indices):
        if self._state_readonly:
            raise NotImplementedError('Cannot set readonly environment state')
        else:
            if indices is None:
                assert states.shape == self._states.shape
                self._states.copy_(states)
            else:
                assert states.shape == (len(indices), self._states.shape[1])
                self._states[indices] = states


class ProductEnv(StatefulBatchedEnv):
    def __init__(self, envs):
        super().__init__(envs[0], len(envs))
        self.envs = envs

    def _reset_index(self, index):
        return self.envs[index].reset()

    def _step(self, actions):
        next_states, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            next_state, reward, done, info = env.step(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return torch.stack(next_states), torch.tensor(rewards), torch.tensor(dones), infos

    def __repr__(self):
        return f'Batch<{self.n_envs}x{self.proto_env}>'


class BatchWrapper(BaseBatchedEnv):
    def __init__(self, env):
        assert isinstance(env, BaseBatchedEnv)
        self.env = env
        super().__init__(env.proto_env, env.n_envs, max_episode_steps=env._max_episode_steps)

    def _reset_index(self, index):
        raise NotImplementedError

    def partial_reset(self, indices):
        return self.env.partial_reset(indices)

    def step(self, actions):
        return self.env.step(actions)

    def _step(self, actions):
        raise NotImplementedError

    def get_states(self):
        return self.env.get_states()

    def set_states(self, states, indices):
        self.env.set_states(states, indices)


class UnbatchEnv(gym.Wrapper):
    def __init__(self, batched_env):
        assert isinstance(batched_env, BaseBatchedEnv)
        assert batched_env.n_envs == 1, 'Can only unbatch env with n_envs = 1'
        super().__init__(batched_env)
        self._max_episode_steps = batched_env._max_episode_steps

    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        states, rewards, dones, infos = self.env.step(action.unsqueeze(0))
        return states[0], float(rewards[0]), bool(dones[0]), {k: v[0] for k, v in infos.items()}

    def set_state(self, state):
        self.env.set_states(state.unsqueeze(0))
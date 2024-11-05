import torch

from force.data import TransitionBuffer
from force.env import TorchWrapper
from force.policies import Policy


class Runner:
    def __init__(self, env, log=None):
        assert isinstance(env, TorchWrapper)
        self._env = env
        self.log = log
        self._samples_taken = 0
        self._returns = []
        self.reset()

    @property
    def samples_taken(self):
        return self._samples_taken

    def get_returns(self, clear=True):
        returns = list(self._returns)
        if clear:
            self._returns.clear()
        return returns

    def reset(self):
        self._last_obs, _ = self._env.reset()
        self._t = 0
        self._total_reward = 0

    def run(self, policy: Policy, num_steps: int, eval: bool = False):
        # Create buffer in which to store the samples
        buffer = TransitionBuffer(
            self._env.observation_space,
            self._env.action_space,
            capacity=num_steps,
            device=self._env.device
        )

        for _ in range(num_steps):
            with torch.no_grad():
                action = policy.act1(self._last_obs, eval)
                # print(actions[0].cpu().numpy())
            next_obs, reward, terminal, truncated, info = self._env.step(action)
            self._samples_taken += 1
            self._t += 1
            self._total_reward += reward

            buffer.append(
                observations=self._last_obs,
                actions=action,
                next_observations=next_obs,
                rewards=reward,
                terminals=terminal,
                truncateds=truncated
            )

            if terminal or truncated:
                self.log(f'Completed episode of length {self._t}, return {self._total_reward:.2f}')
                self._returns.append(self._total_reward)
                self.reset()
            else:
                self._last_obs = next_obs

        return buffer
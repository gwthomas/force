from abc import abstractmethod
from collections import defaultdict

import torch

from force import defaults
from force.data import TransitionBuffer
from force.log import get_global_log
from force.nn import ConfigurableModule
from force.policies import Policy
from force.util import prefix_dict_keys, pymean


class Agent(ConfigurableModule, Policy):
    """Base class for RL algorithm implementations. An Agent
        - is configurable
        - contains PyTorch modules
        - makes decisions
        - learns from data
    """

    class Config(ConfigurableModule.Config):
        discount = defaults.DISCOUNT

    def __init__(self, cfg, obs_space, act_space, device=None):
        ConfigurableModule.__init__(self, cfg, device=device)
        self.obs_space = obs_space
        self.act_space = act_space
        self.log = get_global_log()
        self.train_diagnostics = defaultdict(list)

    def reset_train_diagnostics(self) -> dict:
        results = {k: pymean(v) for k, v in self.train_diagnostics.items()}
        self.train_diagnostics = defaultdict(list)
        return results

    # Optional override
    def additional_diagnostics(self) -> dict:
        return {}

    def process_initial_data(self, data: TransitionBuffer):
        raise NotImplementedError

    @abstractmethod
    def update(self, new_samples: TransitionBuffer, counters: dict):
        raise NotImplementedError


class BufferedAgent(Agent):
    """Base class for agents that use a replay buffer rather than operating
    directly on the most recent batch of data."""

    class Config(Agent.Config):
        buffer_device = 'cpu'
        buffer_capacity = 10**6
        batch_size = defaults.BATCH_SIZE

    def __init__(self, cfg, obs_space, act_space,
                 replay_buffer=None, device=None):
        super().__init__(cfg, obs_space, act_space, device=device)

        if replay_buffer is None:
            replay_buffer = TransitionBuffer(
                obs_space, act_space,
                self.cfg.buffer_capacity,
                device=torch.device(self.cfg.buffer_device)
            )
        else:
            # Check compatibility with config
            assert replay_buffer.device == cfg.buffer_device
            assert replay_buffer.capacity == cfg.buffer_capacity
        self.replay_buffer = replay_buffer

    def process_initial_data(self, data: TransitionBuffer):
        self.replay_buffer.extend(**data.get(as_dict=True))

    def get_minibatch(self):
        # Sample minibatch from replay buffer, move to agent's device if needed
        minibatch = self.replay_buffer.sample(self.cfg.batch_size)
        if self.device != self.replay_buffer.device:
            minibatch = {k: v.to(self.device) for k, v in minibatch.items()}
        return minibatch

    def update_with_minibatch(self, batch: dict, counters: dict):
        raise NotImplementedError

    def _update(self, counters: dict):
        self.update_with_minibatch(self.get_minibatch(), counters)

    def update(self, new_samples: TransitionBuffer, counters: dict):
        # Add new data to replay buffer
        self.replay_buffer.extend(**new_samples.get(as_dict=True))

        # Actually perform update
        self._update(counters)

    def additional_diagnostics(self) -> dict:
        return {
            'buffer_size': len(self.replay_buffer)
        }
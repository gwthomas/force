from abc import abstractmethod
from collections import defaultdict

import numpy as np
import torch

from force import defaults
from force.config import Optional
from force.data import TransitionBuffer
from force.log import get_global_log
from force.nn import ConfigurableModule
from force.nn.module import LambdaModule
from force.nn.util import get_device
from force.policies import BasePolicy, PolicyMode
from force.util import pymean


class BaseAgent(ConfigurableModule, BasePolicy):
    """Base class for RL algorithm implementations. An Agent
        - is configurable
        - contains PyTorch modules
        - makes decisions
        - learns from data
    """

    class Config(ConfigurableModule.Config):
        discount = defaults.DISCOUNT

    def __init__(self, cfg, env_info, device=None):
        ConfigurableModule.__init__(self, cfg, device=device)
        self.env_info = env_info
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
    def train(self, new_samples: TransitionBuffer, counters: dict):
        raise NotImplementedError
    
    def export_policy(self, mode: PolicyMode,
                      path: str | None = None) -> torch.export.ExportedProgram:
        return LambdaModule(
            self.functional(mode), 
            named_parameters=dict(self.named_parameters()),
            input_shape=self.env_info.observation_shape
        ).export(path)
    

class BufferedAgent(BaseAgent):
    """Base class for agents that use a replay buffer rather than operating
    directly on the most recent batch of data."""

    class Config(BaseAgent.Config):
        buffer_device = Optional(str)
        buffer_capacity = 10**6
        batch_size = defaults.BATCH_SIZE
        updates_per_train = 1

    def __init__(self, cfg, env_info,
                 replay_buffer=None, device=None):
        super().__init__(cfg, env_info, device=device)

        buffer_device = get_device(cfg.buffer_device)
        if replay_buffer is None:
            replay_buffer = TransitionBuffer(
                env_info,
                self.cfg.buffer_capacity,
                device=torch.device(buffer_device)
            )
        else:
            # Check compatibility with config
            assert replay_buffer.capacity == cfg.buffer_capacity
            replay_buffer.to(buffer_device)
        self.replay_buffer = replay_buffer

    def process_initial_data(self, data):
        self.add_data(data)

    def add_data(self, data):
        if isinstance(data, TransitionBuffer):
            data_dict = data.get(as_dict=True)
        elif isinstance(data, dict):
            data_dict = data
        else:
            raise ValueError('Data should be dict or buffer')
        self.replay_buffer.extend(**data_dict)

    def get_minibatch(self):
        # Sample minibatch from replay buffer, move to agent's device if needed
        minibatch = self.replay_buffer.sample(self.cfg.batch_size)
        if self.device != self.replay_buffer.device:
            minibatch = {k: v.to(self.device) for k, v in minibatch.items()}
        return minibatch

    def update_with_minibatch(self, batch: dict, counters: dict):
        raise NotImplementedError

    def update(self, counters: dict):
        self.update_with_minibatch(self.get_minibatch(), counters)

    def train(self, new_samples: TransitionBuffer, counters: dict):
        self.add_data(new_samples)
        for _ in range(self.cfg.updates_per_train):
            self.update(counters)

    def additional_diagnostics(self) -> dict:
        return {
            'buffer_size': len(self.replay_buffer)
        }
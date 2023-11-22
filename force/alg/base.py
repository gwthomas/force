from abc import ABC, abstractmethod
from collections import defaultdict
import time
from typing import Callable

import torch
from tqdm import trange

from force import defaults
from force.env.batch import BatchedEnv
from force.log import get_global_log
from force.nn import ConfigurableModule
from force.nn.util import get_device
from force.policies import BasePolicy
from force.sampling import ReplayBuffer, SimpleSampler, sample_episodes_batched
from force.util import prefix_dict_keys, pymean


class IterativeAlgorithm(ABC, ConfigurableModule):
    class Config(ConfigurableModule.Config):
        max_iterations = 1000

    def __init__(self, cfg):
        ConfigurableModule.__init__(self, cfg)
        self.log = get_global_log()
        self.iterations_done = 0

    def get_counters(self) -> dict:
        return {
            'iterations': self.iterations_done
        }

    def pre_iteration(self) -> dict:
        return {}

    def post_iteration(self) -> dict:
        return {}

    @abstractmethod
    def iteration(self) -> dict:
        raise NotImplementedError

    def run(self, summary_writer=None):
        while self.iterations_done < self.cfg.max_iterations:
            pre_info = self.pre_iteration()
            info = self.iteration()
            self.iterations_done += 1
            post_info = self.post_iteration()

            all_info = {**pre_info, **info, **post_info}

            self.log(f'Iteration {self.iterations_done} info:')
            for k, v in all_info.items():
                self.log(f'\t{k}: {v}')
                if summary_writer is not None:
                    summary_writer.add_scalar(k, v, self.iterations_done)


class Agent(ConfigurableModule, BasePolicy):
    """
    Key base class for RL algorithm implementations.
    An Agent
        - is configurable
        - contains PyTorch modules
        - makes decisions
        - learns from data
    """

    class Config(ConfigurableModule.Config):
        discount = defaults.DISCOUNT
        batch_size = defaults.BATCH_SIZE

    def __init__(self, cfg, obs_space, act_space):
        ConfigurableModule.__init__(self, cfg)
        self.obs_space = obs_space
        self.act_space = act_space
        self.log = get_global_log()
        self.train_diagnostics = defaultdict(list)

    def reset_train_diagnostics(self) -> dict:
        results = {k: pymean(v) for k, v in self.train_diagnostics.items()}
        self.train_diagnostics = defaultdict(list)
        return results

    # Optional override
    def additional_diagnostics(self, data: ReplayBuffer) -> dict:
        return {
            'buffer_size': len(data)
        }

    def update_with_batch(self, batch: dict, counters: dict):
        raise NotImplementedError

    def update(self, buffer: ReplayBuffer, counters: dict):
        self.update_with_batch(buffer.sample(self.cfg.batch_size), counters)


class RLAlgorithm(IterativeAlgorithm):
    class Config(IterativeAlgorithm.Config):
        num_eval_episodes = 10

    def __init__(self, cfg, env_factory: Callable, agent: Agent):
        super().__init__(cfg)
        self.env_factory = env_factory
        self.env = env_factory()
        self.agent = agent
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.eval_env = BatchedEnv([env_factory() for _ in range(cfg.num_eval_episodes)])

        # Set this to additionally report normalized returns
        self.return_normalizer = None

    def evaluate(self) -> dict:
        self.log('Evaluating...')
        t_start = time.time()
        eval_episodes = sample_episodes_batched(
            self.eval_env, self.agent, self.cfg.num_eval_episodes, eval=True
        )
        t_end = time.time()
        returns = torch.tensor([ep.get('rewards').sum() for ep in eval_episodes])
        lengths = torch.tensor([len(ep) for ep in eval_episodes])
        eval_stats = {
            'return_mean': returns.mean().item(),
            'return_std': returns.std().item(),
            'return_min': returns.min().item(),
            'return_max': returns.max().item(),
            'length_mean': lengths.float().mean().item(),
            'length_std': lengths.float().std().item(),
            'length_min': lengths.min().item(),
            'length_max': lengths.max().item(),
            'time': t_end - t_start
        }
        if self.return_normalizer is not None:
            normalized_returns = torch.tensor([self.return_normalizer(r) for r in returns])
            eval_stats.update({
                'normalized_return_mean': normalized_returns.mean().item(),
                'normalized_return_std': normalized_returns.std().item(),
                'normalized_return_min': normalized_returns.min().item(),
                'normalized_return_max': normalized_returns.max().item()
            })
        return eval_stats


class BufferedRLAlgorithm(RLAlgorithm):
    class Config(RLAlgorithm.Config):
        buffer_capacity = 10**6
        buffer_min = 1000 # will not update if buffer smaller than this
        updates_per_iter = 1000
        steps_per_update = 1     # set this to 0 for offline algorithms

    def __init__(self, cfg, env_factory, agent,
                 device=None, initial_data=None):
        super().__init__(cfg, env_factory, agent)
        device = get_device(device)

        self.sampler = SimpleSampler(self.env)
        self.buffer = ReplayBuffer(
            self.env.observation_space,
            self.env.action_space,
            cfg.buffer_capacity,
            device=device
        )
        if isinstance(initial_data, ReplayBuffer):
            self.buffer.extend(initial_data.get(as_dict=True))
        elif isinstance(initial_data, dict):
            self.buffer.extend(initial_data)
        elif initial_data is None:
            pass
        else:
            raise ValueError('Initial data should be ReplayBuffer or dict')

        self.to(device)
        self.updates_done = 0

    def get_counters(self) -> dict:
        return {
            'updates': self.updates_done,
            'env_steps': self.sampler.steps_done,
            **super().get_counters()
        }

    def iteration(self) -> dict:
        self.log('Training...')
        for _ in trange(self.cfg.updates_per_iter):
            # Collect data
            self.sampler.run(
                self.agent, self.cfg.steps_per_update,
                buffer=self.buffer, eval=False
            )

            # Update (if we have enough data)
            if len(self.buffer) > self.cfg.buffer_min:
                self.agent.update(self.buffer, self.get_counters())
                self.updates_done += 1

        info = {
            **prefix_dict_keys('counters', self.get_counters()),
            **prefix_dict_keys('eval', self.evaluate()),
            **prefix_dict_keys('train', self.agent.reset_train_diagnostics()),
            **prefix_dict_keys('info', self.agent.additional_diagnostics(self.buffer)),
        }
        sampled_returns = self.sampler.get_returns()
        if sampled_returns:
            info['sampling/return'] = pymean(sampled_returns)
        return info
from abc import abstractmethod
import time

import h5py
import torch
from tqdm import trange

from force.alg.agent import Agent
from force.config import Field
from force.data import TransitionBuffer
from force.env import EnvConfig, VectorEnvConfig, get_env
from force.experiment.epochal import EpochalExperiment
from force.policies import UniformPolicy
from force.sampling import Runner, sample_episode
from force.util import prefix_dict_keys, pymean


class RLExperiment(EpochalExperiment):
    class Config(EpochalExperiment.Config):
        env = EnvConfig
        eval_env = Field(EnvConfig, required=False)
        updates_per_epoch = int
        initial_steps = 0
        steps_per_update = 1
        num_eval_episodes = 10
        save_eval_episodes = False

    def __init__(self, cfg):
        super().__init__(cfg)

        self.env = get_env(cfg.env)
        self.log(f'Env: {repr(self.env)}')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.eval_env = get_env(cfg.eval_env if cfg.eval_env is not None else cfg.env)
        if cfg.save_eval_episodes:
            self.eval_path = self.log_dir/'evals.hdf5'

        self.runner = Runner(self.env, log=self.log)

        self.agent = self.create_agent()
        self.log(f'Agent: {self.agent}')

        # Set this to a callable to additionally report normalized returns
        self.return_normalizer = None

        # Counter
        self.updates_done = 0

        # Optionally initialize the agent with transitions.
        # By default these are sampled via a uniform policy, but one can also
        # override get_initial_data to supply data from another source.
        initial_data = self.get_initial_data()
        if initial_data is not None:
            self.agent.process_initial_data(initial_data)

    def get_counters(self) -> dict:
        return {
            'updates': self.updates_done,
            'env_samples': self.runner.samples_taken,
            **super().get_counters()
        }

    @abstractmethod
    def create_agent(self) -> Agent:
        raise NotImplementedError

    def get_initial_data(self):
        steps = self.cfg.initial_steps
        if steps > 0:
            policy = UniformPolicy(self.action_space)
            return self.runner.run(policy, steps, eval=False)
        else:
            return None

    def epoch(self) -> dict:
        self.log('Training...')
        for _ in trange(self.cfg.updates_per_epoch):
            batch = self.runner.run(
                self.agent, self.cfg.steps_per_update, eval=False
            )
            self.agent.update(batch, self.get_counters())
            self.updates_done += 1

        info = {
            **prefix_dict_keys('counters', self.get_counters()),
            **prefix_dict_keys('train', self.agent.reset_train_diagnostics()),
            **prefix_dict_keys('info', self.agent.additional_diagnostics()),
        }
        sampled_returns = self.runner.get_returns()
        if sampled_returns:
            info['sampling/return_mean'] = pymean(sampled_returns)
        return info

    def post_epoch(self) -> dict:
        return prefix_dict_keys('eval', self.evaluate())

    def evaluate(self) -> dict:
        self.log('Evaluating...')
        t_start = time.time()
        eval_episodes = [
            sample_episode(self.eval_env, self.agent, eval=True)
            for _ in range(self.cfg.num_eval_episodes)
        ]
        t_end = time.time()

        # Optionally save episodes
        if self.cfg.save_eval_episodes:
            with h5py.File(self.eval_path, 'a') as f:
                for i, ep in enumerate(eval_episodes):
                    ep.save_to_file(f, prefix=f'iteration_{self.iterations_done}/episode_{i}')

        # Collect statistics
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
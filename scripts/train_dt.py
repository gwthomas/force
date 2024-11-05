import time

import torch
import torch.multiprocessing as mp
from tqdm import trange

from force.config import Field
from force.data.transition_buffer import TransitionBuffer
from force.defaults import DEVICE
from force.env import EnvConfig, get_env
from force.experiment import EpochalExperiment
from force.nn.models.decision_transformer import DecisionTransformer, DTSampler, evaluate_dt
from force.nn.util import torchify, batch_iterator, quartiles
from force.sampling import Minibatcher
from force.util import pymean, prefix_dict_keys


class DTExperiment(EpochalExperiment):
    class Config(EpochalExperiment.Config):
        algorithm = 'DT'
        env = EnvConfig
        target_return = Field(float, required=False)
        dt = DecisionTransformer.Config
        batch_size = 64
        updates_per_epoch = 1000
        num_eval_episodes = 10
        data_device = 'cpu'
        num_processes = 1

    def __init__(self, cfg):
        super().__init__(cfg)

        self.env = env = self.env_factory()
        self.eval_envs = [self.env_factory() for _ in range(cfg.num_eval_episodes)]
        self.T = env.spec.max_episode_steps

        self.dt = DecisionTransformer(cfg.dt, env.observation_space, env.action_space)
        self.dt.to(DEVICE)

        self.data_device = torch.device(cfg.data_device)
        self.load_data()

        # Initialize sampler
        samplers = [
            DTSampler(self.trajectories, cfg.dt.history_len, self.data_device)
            for _ in range(cfg.num_processes)
        ]
        self.batch_sampler = Minibatcher(samplers, cfg.batch_size)

        # Use config-specified target return if it exists,
        # otherwise the empirical max return in the dataset
        self.target_return = cfg.target_return if cfg.target_return is not None else \
                             max(self.dataset_returns)

    def env_factory(self):
        import d4rl
        return get_env(self.cfg.env)

    def load_data(self):
        self.log('Loading data...')
        data = self.env.get_dataset()
        data = {
            k: torchify(data[k], device=self.data_device)
            for k in (
                'observations', 'actions', 'next_observations', 'rewards',
                'terminals', 'timeouts'
            )
        }
        data['truncateds'] = data.pop('timeouts')   # relabel key
        buffer = TransitionBuffer(self.env.observation_space, self.env.action_space, len(data['actions']))
        buffer.extend(**data)
        self.trajectories = buffer.separate_into_trajectories()
        self.dataset_returns = [traj.get('rewards').sum().item() for traj in self.trajectories]
        self.log(f'Dataset contains {len(self.trajectories)} trajectories')
        self.log(f'Return quartiles: {quartiles(self.dataset_returns)}')

    def epoch(self):
        losses = []
        for _ in trange(self.cfg.updates_per_epoch):
            batch = self.batch_sampler.sample()
            loss = self.dt.update(batch)
            losses.append(loss)

        return {
            'initial_loss': losses[0],
            'final_loss': losses[-1],
            'mean_loss': pymean(losses)
        }

    def post_epoch(self) -> dict:
        return prefix_dict_keys('eval', self.evaluate())

    def evaluate(self):
        self.log(f'Evaluating with target return {self.target_return}...')
        t_start = time.time()
        returns = evaluate_dt(self.eval_envs, self.dt, self.target_return)
        t_end = time.time()

        # Collect statistics
        eval_stats = {
            'return_mean': returns.mean().item(),
            'return_std': returns.std().item(),
            'return_min': returns.min().item(),
            'return_max': returns.max().item(),
            'time': t_end - t_start
        }
        # if self.return_normalizer is not None:
        #     normalized_returns = torch.tensor([self.return_normalizer(r) for r in returns])
        #     eval_stats.update({
        #         'normalized_return_mean': normalized_returns.mean().item(),
        #         'normalized_return_std': normalized_returns.std().item(),
        #         'normalized_return_min': normalized_returns.min().item(),
        #         'normalized_return_max': normalized_returns.max().item()
        #     })
        return eval_stats

    def cleanup(self):
        self.log('Stopping sampler...')
        self.batch_sampler.stop()
        self.log('Sampler stopped.')

        super().cleanup()


if __name__ == '__main__':
    # MPS seems to require fork for multiprocessing, but CUDA doesn't work with fork
    mp.set_start_method('fork' if torch.backends.mps.is_available() else 'spawn')

    # Default sharing strategy gave error
    mp.set_sharing_strategy('file_system')

    DTExperiment.main()
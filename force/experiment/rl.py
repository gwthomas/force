from abc import abstractmethod
import time

import torch
from tqdm import trange

from force.alg.agent import BaseAgent
from force.env import GymEnv
from force.experiment.epochal import EpochalExperiment
from force.policies import PolicyMode, UniformPolicy
from force.sampling import Runner, rollout
from force.scripts.rollout import rollout_cmd
from force.util import prefix_dict_keys, stats_dict, pymean


class RLExperiment(EpochalExperiment):
    class Config(EpochalExperiment.Config):
        env_name = str
        trains_per_epoch = int
        initial_steps = 0
        steps_per_train = 1
        num_eval_episodes = 10
        use_async_eval = False

    def __init__(self, cfg):
        super().__init__(cfg)

        self.env = GymEnv(cfg.env_name, device=self.device)
        self.log(f'Env: {repr(self.env)}')

        if not self.cfg.use_async_eval:
            # Create envs once ahead of time
            self.eval_envs = [
                GymEnv(cfg.env_name, device=self.device)
                for _ in range(cfg.num_eval_episodes)
            ]

        self.runner = Runner(self.env, log=self._log)

        self.agent = self.create_agent()
        self.log(f'Agent: {self.agent}')
        self.register_checkpointables(agent=self.agent)

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
    def create_agent(self) -> BaseAgent:
        raise NotImplementedError

    def get_initial_data(self):
        steps = self.cfg.initial_steps
        if steps > 0:
            policy = UniformPolicy(self.env.action_space)
            return self.runner.run(policy.functional(PolicyMode.EXPLORE), steps)
        else:
            return None

    def epoch(self) -> dict:
        self.log('Training...')
        for _ in trange(self.cfg.trains_per_epoch):
            batch = self.runner.run(
                self.agent.functional(PolicyMode.EXPLORE), self.cfg.steps_per_train,
            )
            self.agent.train(batch, self.get_counters())
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
        eval_rollout_dir = self._log_dir/'eval_rollouts'
        eval_rollout_dir.mkdir(exist_ok=True)
        eval_rollout_path = eval_rollout_dir/f'epoch_{self._epochs_done}.hdf5'

        if self.cfg.use_async_eval:
            policy_path = self._log_dir/f'policy_{self._epochs_done}.pt2'
            self.log(f'Exporting policy to {policy_path} for async eval')
            self.agent.export_policy(PolicyMode.EVAL, path=policy_path)
            self.job_manager.enqueue(
                name=f'eval_{self._epochs_done}',
                cmd=rollout_cmd(
                    self.cfg.env_name, policy_path, self.cfg.num_eval_episodes,
                    out_path=eval_rollout_path,
                    verbose=True
                )
            )
            return {}
        else:
            self.log(f'Evaluating policy.')
            t_start = time.time()
            eval_policy = self.agent.functional(PolicyMode.EVAL)
            eval_trajs = rollout(self.eval_envs, eval_policy,
                                 out_path=eval_rollout_path)
            t_end = time.time()

            # Collect statistics
            returns = torch.tensor([ep.get('rewards').sum() for ep in eval_trajs])
            lengths = torch.tensor([len(ep) for ep in eval_trajs])
            eval_stats = {
                **prefix_dict_keys('return', stats_dict(returns)),
                **prefix_dict_keys('length', stats_dict(lengths)),
                'time': t_end - t_start
            }
            if self.return_normalizer is not None:
                normalized_return_stats = stats_dict([self.return_normalizer(r) for r in returns])
                eval_stats.update(prefix_dict_keys('normalized_return', normalized_return_stats))
            return eval_stats
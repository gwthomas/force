from abc import ABC, abstractmethod
from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
import random
import signal
import traceback

import torch
from torch.utils.tensorboard import SummaryWriter

from force.alg.base import IterativeAlgorithm
from force.config import BaseConfig, Configurable, Field
from force.log import Log, set_global_log
from force.util import set_seed, random_string, try_parse, time_since


class Experiment(ABC, Configurable):
    """
    The intended usage is to subclass Experiment, defining a Config class
    (which should subclass Experiment.Config) and a setup() method that creates
    an IterativeAlgorithm to be run.
    """
    class Config(BaseConfig):
        root_dir = Field(str)
        domain = Field(str)
        algorithm = Field(str)
        run_id = Field(str, required=False)
        seed = Field(int, required=False)
        debug = False

    @abstractmethod
    def setup(self, cfg) -> IterativeAlgorithm:
        raise NotImplementedError

    def _write_status(self, status: str):
        assert isinstance(status, str)
        (self.log_dir/'status.txt').write_text(status)
        self.log(status)

    def run(self, cfg):
        start_t = datetime.now()

        # If no run_id is specified, generate one
        if cfg.run_id is None:
            start_str = start_t.strftime('%y-%m-%d_%H.%M.%S')
            rand_str = random_string(4, include_uppercase=False, include_digits=False)
            cfg.run_id = f'{start_str}_{rand_str}'

        # Random seed
        if cfg.seed is None:
            random.seed()
            cfg.seed = random.randrange(100)
        set_seed(cfg.seed)

        # Set up directory and logs
        root_log_dir = Path(cfg.root_dir) / ('debug_logs' if cfg.debug else 'logs')
        domain_algorithm_dir = root_log_dir / cfg.domain / cfg.algorithm
        domain_algorithm_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir = domain_algorithm_dir / cfg.run_id
        self.log_dir.mkdir(exist_ok=True)
        self.log = Log(self.log_dir / 'log.txt')
        set_global_log(self.log)
        self.summary_writer = SummaryWriter(log_dir=self.log_dir)

        # Dump config to file
        config_path = self.log_dir/'config.json'
        with config_path.open('w') as f:
            json.dump(cfg.vars_recursive(), f, indent=2)
        self.log(f'Wrote config to {config_path}')

        # Setup
        torch.set_num_threads(1)
        signal.signal(signal.SIGTERM, lambda signal, frame: self._write_status(f'KILLED (SIGTERM)'))

        try:
            algorithm = self.setup(cfg)
            assert isinstance(algorithm, IterativeAlgorithm), \
                'Experiment.setup() should return an IterativeAlgorithm'
            self._write_status('STARTED')
            algorithm.run(summary_writer=self.summary_writer)
        except KeyboardInterrupt:
            self._write_status('KILLED')
        except Exception:
            exc_str = traceback.format_exc()
            self._write_status(f'ERROR after {time_since(start_t)}:\n{exc_str}')
        else:
            # Experiment finished running without error
            self._write_status(f'DONE after {time_since(start_t)}')

        # Clean up
        self.log.close()
        self.summary_writer.close()

    @classmethod
    def main(cls):
        cfg = cls.Config()
        assert isinstance(cfg, Experiment.Config)

        parser = ArgumentParser()
        parser.add_argument('-c', '--config', default=[], action='append')
        parser.add_argument('-s', '--set', default=[], action='append', nargs=2)
        parser.add_argument('--root-dir', default=None)
        parser.add_argument('--domain', default=None)
        parser.add_argument('--algorithm', default=None)
        parser.add_argument('--run-id', default=None)
        parser.add_argument('--seed', type=int, default=None)
        parser.add_argument('--debug', action='store_true')
        args = parser.parse_args()

        # Load config from file
        for cfg_path in args.config:
            with Path(cfg_path).open('r') as f:
                cfgd = json.load(f)
            assert isinstance(cfgd, dict)
            cfg.update(cfgd)

        # Override specific arguments
        for key, value in args.set:
            cfg.set(key, try_parse(value))

        # Optionally override Experiment args
        for key in ['root_dir', 'domain', 'algorithm', 'run_id', 'seed']:
            val = getattr(args, key)
            if val is not None:
                setattr(cfg, key, val)
        if args.debug:
            cfg.debug = True

        # Ensure all required arguments have been set
        try:
            cfg.resolve()
        except ValueError as e:
            print('Failed to verify config:')
            print(cfg)
            raise e

        experiment = cls(cfg)
        experiment.run(cfg)
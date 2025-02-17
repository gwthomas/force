from abc import ABC, abstractmethod
from argparse import ArgumentParser
from datetime import datetime
import os
from pathlib import Path
import random
import shutil
import signal
import traceback

from torch.utils.tensorboard import SummaryWriter

from force.defaults import DATETIME_FORMAT, TORCH_NUM_THREADS, MAX_SEED
from force.config import Configurable, Optional
from force.parallel.jobs import JobManager
from force.log import Log, set_global_log
from force.nn.util import get_device
from force.util import set_seed, random_string, load_configd, update_cfgd, try_parse, time_since


class Experiment(ABC, Configurable):
    """
    The intended usage is to subclass Experiment, defining a Config class
    (which should subclass Experiment.Config) and overriding the run() method.
    """
    class Config(Configurable.Config):
        project_dir = Optional(str)
        run_id = Optional(str)
        seed = Optional(int)
        device = Optional(str)
        debug = False

    def __init__(self, cfg):
        Configurable.__init__(self, cfg)

        # Set up directory and logs
        if cfg.project_dir is None:
            cfg.project_dir = os.getcwd()
        project_dir = Path(cfg.project_dir)
        project_log_dir = project_dir/'logs'
        project_log_dir.mkdir(exist_ok=True, parents=True)
        self._log_dir = project_log_dir/cfg.run_id
        if self._log_dir.is_dir() and cfg.debug:
            print('Deleting existing log dir')
            shutil.rmtree(self._log_dir)
        self._log_dir.mkdir(exist_ok=True)
        self._log = Log(self._log_dir/'log.txt')
        self.log(f'Log dir: {self._log_dir}')
        set_global_log(self._log)
        self._summary_writer = SummaryWriter(log_dir=self._log_dir)

        # Dump config to file
        config_path = self._log_dir/'config.yaml'
        config_path.write_text(cfg.to_yaml())

        # Device
        self.device = get_device(cfg.device)

        # Job manager
        self.job_manager = JobManager(self._log)

    def log(self, msg: str):
        self._log.write(msg)

    def _write_status(self, status: str):
        assert isinstance(status, str)
        assert self._log_dir.is_dir()
        (self._log_dir/'status.txt').write_text(status)
        self.log(status)

    @abstractmethod
    def run(self):
        raise NotImplementedError

    # This can be overridden, but don't forget to call super().cleanup()
    def cleanup(self):
        self._log.close()
        self._summary_writer.close()

    def managed_run(self):
        start_t = datetime.now()
        self._write_status('STARTED')

        try:
            self.run()
        except KeyboardInterrupt:
            self._write_status('KILLED')
        except Exception:
            exc_str = traceback.format_exc()
            self._write_status(f'ERROR after {time_since(start_t)}:\n{exc_str}')
        else:
            # Experiment finished running without error
            self._write_status(f'DONE after {time_since(start_t)}')

        self.cleanup()

    @classmethod
    def main(cls):
        cfg_class = cls.Config
        assert issubclass(cfg_class, Experiment.Config)
        cfg = cfg_class()

        parser = ArgumentParser()
        parser.add_argument('-c', '--config', default=[], action='append')
        parser.add_argument('-s', '--set', default=[], action='append', nargs=2)
        parser.add_argument('--project', default=None)
        parser.add_argument('--domain', default=None)
        parser.add_argument('--algorithm', default=None)
        parser.add_argument('--run-id', default=None)
        parser.add_argument('--seed', type=int, default=None)
        parser.add_argument('--debug', action='store_true')
        args = parser.parse_args()

        # Load config(s) from file and combine them before passing to update().
        # This is done in a convoluted way to improve handling of TaggedUnions.
        cfgd = {}
        for cfg_path in args.config:
            update_cfgd(cfgd, load_configd(cfg_path))
        cfg.update(cfgd)

        # Override specific arguments
        for key, value in args.set:
            cfg.set(key, try_parse(value))

        # Optionally override Experiment args
        for key in ['project', 'domain', 'algorithm', 'run_id', 'seed']:
            val = getattr(args, key)
            if val is not None:
                cfg.set(key, val)

        debug = args.debug
        if debug:
            cfg.set('debug', True)

        if cfg.get('run_id') is None:
            if debug:
                run_id = 'debug'
            else:
                # Generate run ID with current date+time and a random string
                start_t = datetime.now()
                start_str = start_t.strftime(DATETIME_FORMAT)
                rand_str = random_string(4, include_uppercase=False, include_digits=False)
                run_id = f'{start_str}_{rand_str}'
            cfg.set('run_id', run_id)

        # Random seed
        if cfg.get('seed') is None:
            random.seed()
            cfg.set('seed', random.randrange(MAX_SEED))
        set_seed(cfg.get('seed'))

        # Ensure all required arguments have been set
        try:
            cfg.resolve()
        except Exception as e:
            print(f'Failed to verify config:\n{cfg}')
            print(f'Key: {".".join(e.key_list)}')
            print(f'Error: {e}')
            return

        # Run experiment
        experiment = cls(cfg)
        experiment.managed_run()
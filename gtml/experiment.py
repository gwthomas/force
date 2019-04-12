import os

import torch

from gtml.constants import EXPERIMENTS_DIR


class Data(dict):
    def append(self, key, value):
        if key not in self:
            self[key] = []
        self[key].append(value)


class Experiment:
    def __init__(self, name, serializables={}, ensure_new=False, ensure_exists=False):
        self.name = name
        self.serializables = serializables
        self.data = Data()
        self.dir = os.path.join(EXPERIMENTS_DIR, name)

        if os.path.isfile(self.log_path): # Check for existing data
            print('Found existing experiment at {}'.format(self.dir))
            if ensure_new:
                print('ensure_new is True; exiting')
                exit(1)

            self.log_file = open(self.log_path, 'a')
        else:
            print('No experiment exists at {}'.format(self.dir))
            if ensure_exists:
                print('ensure_exists is True; exiting')
                exit(1)

            print('Creating experiment')
            if not os.path.isdir(self.dir):
                os.makedirs(self.dir)
            self.log_file = open(self.log_path, 'w')

    @property
    def log_path(self):
        return os.path.join(self.dir, 'log.txt')

    def data_path(self, index):
        return os.path.join(self.dir, 'data_{}.pt'.format(index))

    def checkpoint_path(self, index):
        return os.path.join(self.dir, 'checkpoint_{}.pt'.format(index))

    def log(self, format_string, *args):
        output = format_string.format(*args)
        print(output)
        self.log_file.write(output + '\n')

    def load(self, index, raise_on_missing=True, raise_on_extra=False):
        self.data = torch.load(self.data_path(index))
        checkpoint = torch.load(self.checkpoint_path(index))

        available = set(checkpoint.keys())
        requested = set(self.serializables.keys())
        loaded = set()

        missing = requested - available
        if missing:
            self.log('WARNING: the following states were requested but not available:', list(missing))
            if raise_on_missing:
                raise RuntimeError('raise_on_missing triggered')

        extra = available - requested
        if extra:
            self.log('WARNING: the following states were available but not requested:', list(extra))
            if raise_on_extra:
                raise RuntimeError('raise_on_extra triggered')

        for name in requested:
            if name in available:
                self.serializables[name].load_state_dict(checkpoint[name])

        self.log('Load completed successfully')

    def save(self, index):
        checkpoint = {name: obj.state_dict() for name, obj in self.serializables.items()}
        for obj, path_fn in [(self.data, self.data_path),
                             (checkpoint, self.checkpoint_path)]:
            path = path_fn(index)
            torch.save(obj, path)

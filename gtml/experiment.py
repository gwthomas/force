import glob
import os
import string

import torch

from gtml.constants import EXPERIMENTS_DIR, DEVICE


class Data(dict):
    def append(self, key, value):
        if key not in self:
            self[key] = []
        self[key].append(value)


class Experiment:
    def __init__(self, name, exp_dir=EXPERIMENTS_DIR, serializables=None,
                 ensure_new=False, ensure_exists=False, verbose=True):
        self.name = name
        self.serializables = {} if serializables is None else serializables
        self.data = Data()
        self.dir = os.path.join(exp_dir, name)
        self.verbose = verbose

        if os.path.isfile(self.log_path): # Check for existing data
            if self.verbose:
                print('Found existing experiment at {}'.format(self.dir))
            if ensure_new:
                print('ensure_new is True; exiting')
                exit(1)

            self.log_file = open(self.log_path, 'a')
        else:
            if self.verbose:
                print('No experiment exists at {}'.format(self.dir))
            if ensure_exists:
                print('ensure_exists is True; exiting')
                exit(1)

            print('Creating experiment')
            os.makedirs(self.dir, exist_ok=True)
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

    def register_serializable(self, name, serializable):
        self.serializables[name] = serializable
        
    def list_checkpoints(self):
        checkpoint_indices = []
        for path in glob.glob(self.checkpoint_path('*')):
            filename = os.path.basename(path)
            digits = filename.lstrip('checkpoint_').rstrip('.pt')
            checkpoint_indices.append(int(digits))
        return sorted(checkpoint_indices)

    def load(self, index, load_data=True, load_checkpoint=True,
             raise_on_missing=True, raise_on_extra=False):
        if self.verbose:
            self.log('Attempting to load from checkpoint {}', index)
            
        if load_data:
            self.data = torch.load(self.data_path(index))
            
        if load_checkpoint:
            checkpoint = torch.load(self.checkpoint_path(index),
                                    map_location=DEVICE)

            existing_keys = set(checkpoint.keys())
            requested_keys = set(self.serializables.keys())

            missing = requested_keys - existing_keys
            if missing:
                if self.verbose:
                    self.log('WARNING: the following serializable keys were requested but do not exist in the checkpoint:', list(missing))
                if raise_on_missing:
                    raise RuntimeError('raise_on_missing triggered')

            extra = existing_keys - requested_keys
            if extra:
                if self.verbose:
                    self.log('WARNING: the following serializables exist in the checkpoint but were not requested:', list(extra))
                if raise_on_extra:
                    raise RuntimeError('raise_on_extra triggered')

            for key in requested_keys:
                if key in existing_keys:
                    self.serializables[key].load_state_dict(checkpoint[key])
                    
        if self.verbose:
            self.log('Load successful!')

    # Note: this assumes checkpoints are indexed by integers
    def load_latest(self, **kwargs):
        checkpoint_indices = self.list_checkpoints()
        if len(checkpoint_indices) == 0:
            if self.verbose:
                self.log('No available checkpoints')
            return False
        if self.verbose:
            self.log('Available checkpoint indices: {}', checkpoint_indices)
        latest_index = checkpoint_indices[-1]
        self.load(latest_index, **kwargs)
        return True
    
    def load_other(self, name, index=None, **kwargs):
        load_exp = Experiment(name, serializables=self.serializables,
                              ensure_exists=True)
        if index is None:
            load_exp.load_latest(**kwargs)
        else:
            load_exp.load(index=index, **kwargs)
        self.data = load_exp.data

    def save(self, index):
        checkpoint = {name: obj.state_dict() for name, obj in self.serializables.items()}
        for obj, path_fn in [(self.data, self.data_path),
                             (checkpoint, self.checkpoint_path)]:
            path = path_fn(index)
            torch.save(obj, path)
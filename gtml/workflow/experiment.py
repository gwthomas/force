from collections import defaultdict
from pathlib import Path
import shutil
import sys

import torch

from gtml.constants import DEVICE
import gtml.util as util


class Experiment:
    def __init__(self, base_dir, variant, seed):
        self.base_dir = Path(base_dir)
        self.variant = variant
        self.variant_dir = self.base_dir / 'variants' / self.variant
        self.seed = seed
        self.instance_dir = self.variant_dir / 'seed{}'.format(self.seed)
        self.log_file = None # call instantiate to create this
        self.serializables = {}
        self.data = defaultdict(list)

        if not self.base_dir.is_dir():
            raise ValueError('No experiment family at {}'.format(self.base_dir))

        if not self.variant_dir.is_dir():
            raise ValueError('Variant {} does not exist. Expected path: {}'.format(self.variant, self.variant_dir))

    def instantiate(self, erase_if_exists):
        if self.instance_dir.is_dir():
            if erase_if_exists:
                print('Erasing existing experiment')
                shutil.rmtree(self.instance_dir)
            else:
                raise RuntimeError('Existing experiment found; not erasing')

        self.instance_dir.mkdir()
        self.log_file = open(self.instance_dir / 'log.txt', 'w', buffering=1)

    def data_path(self, index):
        return self.instance_dir / 'data_{}.pt'.format(index)

    def checkpoint_path(self, index):
        return self.instance_dir / 'checkpoint_{}.pt'.format(index)

    def log(self, format_string, *args):
        output = format_string.format(*args)
        print(output)
        self.log_file.write(output + '\n')

    def register_serializables(self, serializables):
        self.serializables.update(serializables)

    def list_checkpoints(self):
        checkpoint_indices = []
        for path in glob.glob(self.checkpoint_path('*')):
            filename = os.path.basename(path)
            digits = filename.lstrip('checkpoint_').rstrip('.pt')
            checkpoint_indices.append(int(digits))
        return sorted(checkpoint_indices)

    def load(self, index, load_data=True, load_checkpoint=True,
             raise_on_missing=True, raise_on_extra=False):
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

    # Note: this assumes checkpoints are indexed by integers
    def load_latest(self, **kwargs):
        checkpoint_indices = self.list_checkpoints()
        if len(checkpoint_indices) == 0:
            raise RuntimeError('No available checkpoints')
        latest_index = checkpoint_indices[-1]
        self.load(latest_index, **kwargs)
        return True

    def load_other(self, name, index=None, **kwargs):
        load_exp = Experiment(name, ensure_exists=True)
        load_exp.register_serializables(self.serializables)
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

    def run(self):
        if self.log_file is None:
            raise RuntimeError('Must instantiate experiment before running')

        sys.path.append(str(self.base_dir.resolve()))
        try:
            import main
        except:
            print('Failed to import main')
            raise

        cfg_path = self.variant_dir / 'config.json'
        cfg = main.config_info.parse_json_file(cfg_path)
        util.set_random_seed(self.seed)
        main.main(self, cfg)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('base_dir', type=str)
    parser.add_argument('variant', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--erase_if_exists', action='store_false')
    args = parser.parse_args()
    exp = Experiment(args.base_dir, args.variant, args.seed)
    exp.instantiate(args.erase_if_exists)
    exp.run()

from collections import defaultdict
import json
from pathlib import Path
import subprocess
import shutil
import sys

import torch

from gtml.constants import DEVICE
import gtml.util as util


class ExperimentFamily:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_dir():
            raise ValueError('No experiment family at {}'.format(self.path))
        self._module = None

    @property
    def module(self):
        if self._module is None:
            sys.path.append(str(self.path.resolve()))
            import main
            self._module = main
        return self._module


class ExperimentVariant:
    def __init__(self, family, name):
        self.family = family
        self.name = name
        self.path = family.path / 'variants' / name

    def setup(self):
        if not self.path.is_dir():
            self.path.mkdir(parents=True)
        spec = self.family.module.variant_specs[self.name]
        spec_text = json.dumps(spec, indent=4)
        spec_path = self.path / 'spec.json'
        spec_path.write_text(spec_text)

    def get_config(self):
        return self.family.config_info.parse_json_file(self.path / 'spec.json')


class ExperimentInstance:
    def __init__(self, variant, seed):
        self.variant = variant
        self.seed = seed
        self.path = variant.path / 'seed{}'.format(seed)
        self.log_file = None # call instantiate to create this
        self.serializables = {}
        self.data = defaultdict(list)

    @property
    def family(self):
        return self.variant.family

    def instantiate(self, erase_if_exists):
        if self.path.is_dir():
            if erase_if_exists:
                print('Erasing existing experiment')
                shutil.rmtree(self.path)
            else:
                raise RuntimeError('Existing experiment found; not erasing')

        self.path.mkdir()
        self.log_file = open(self.path / 'log.txt', 'w', buffering=1)

    def write_batch_script(self):
        cmd = 'srun python {} {} {} --seed {}'.format(
                Path(__file__).resolve(),
                self.family.path.resolve(),
                self.variant.name,
                self.seed
        )

        module = self.family.module
        spec = module.variant_specs[self.variant.name]
        sbatch_args = module.sbatch_args(spec)
        lines = ['#!/bin/bash']
        for key, val in sbatch_args.items():
            lines.append('#SBATCH --{}={}'.format(key, val))
        lines.append(cmd)
        batch_path = self.path / 'batch.sh'
        batch_path.write_text('\n'.join(lines))

    def data_path(self, index):
        return self.path / 'data_{}.pt'.format(index)

    def checkpoint_path(self, index):
        return self.path / 'checkpoint_{}.pt'.format(index)

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
        cfg = self.variant.get_config()
        util.set_random_seed(self.seed)
        self.family.module.main(self, cfg)


def launch(args):
    family = ExperimentFamily(args.base_dir)
    if args.variant not in family.module.variant_specs:
        print('Requested variant', args.variant, 'not found! Existing variants:')
        for existing_variant_name in sorted(family.module.variant_specs.keys()):
            print('\t', existing_variant_name)
        exit()

    variant = ExperimentVariant(family, args.variant)
    variant.setup()
    instance = ExperimentInstance(variant, args.seed)
    instance.instantiate(not args.careful)

    if args.batch:
        print('Writing batch script')
        instance.write_batch_script()
        print('Submitting')
        #completed = subprocess.run([''])
    else:
        instance.run()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('base_dir', type=str)
    parser.add_argument('variant', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--careful', action='store_true')
    launch(parser.parse_args())

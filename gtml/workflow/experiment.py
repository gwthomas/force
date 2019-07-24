import pickle

from collections import defaultdict
import json
from pathlib import Path
import subprocess
import shutil
import sys

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

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
        return self.family.module.config_info.parse_json_file(self.path / 'spec.json')


def _extract_index(checkpoint_path):
    return checkpoint_path.split('/')[-1].lstrip('ckpt-')

class ExperimentInstance:
    def __init__(self, variant, seed):
        self.variant = variant
        self.seed = seed
        self.path = variant.path / 'seed{}'.format(seed)
        self.checkpoint_dir = self.path / 'checkpoints'
        self.data = defaultdict(list)
        self.data_dir = self.path / 'data'
        self.log_file = None # call instantiate to create this

    @property
    def family(self):
        return self.variant.family

    def data_path(self, index):
        return self.data_dir / 'data-{}.pkl'.format(index)

    def instantiate(self, erase_if_exists):
        if self.path.is_dir():
            if erase_if_exists:
                print('Erasing existing experiment')
                shutil.rmtree(self.path)
            else:
                raise RuntimeError('Existing experiment found; not erasing')

        self.path.mkdir()
        self.log_file = open(self.path / 'log.txt', 'w', buffering=1)

    def write_batch_script(self, additional_args={}):
        cmd = 'srun python {} {} {} --seed {}'.format(
                Path(__file__).resolve(),
                self.family.path.resolve(),
                self.variant.name,
                self.seed
        )

        module = self.family.module
        spec = module.variant_specs[self.variant.name]
        sbatch_args = module.sbatch_args(spec)
        sbatch_args.update(additional_args)
        lines = ['#!/bin/bash']
        for key, val in sbatch_args.items():
            lines.append('#SBATCH --{}={}'.format(key, val))
        lines.append(cmd)
        batch_path = self.path / 'batch.sh'
        batch_path.write_text('\n'.join(lines))
        return batch_path

    def log(self, format_string, *args):
        output = format_string.format(*args)
        print(output)
        self.log_file.write(output + '\n')

    def setup_checkpointing(self, trackables, max_to_keep=None):
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.checkpoint = tf.train.Checkpoint(**trackables)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint,
                self.checkpoint_dir,
                max_to_keep=max_to_keep)

    def save(self, index=None):
        saved_path = self.checkpoint_manager.save(index)
        saved_index = _extract_index(saved_path)
        with open(self.data_path(saved_index), 'wb') as f:
            pickle.dump(self.data, f)

    def load(self, index):
        self.checkpoint.restore(self.checkpoint_dir / 'ckpt-{}'.format(index))
        with open(self.data_path(index), 'rb') as f:
            self.data = pickle.load(f)

    def load_latest(self):
        self.load(_extract_index(self.checkpoint_manager.latest_checkpoint))

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
        additional_batch_args = {}
        if args.partition is not None:
            additional_batch_args['partition'] = args.partition
        if args.nodelist is not None:
            additional_batch_args['nodelist'] = args.nodelist
        batch_path = instance.write_batch_script(additional_batch_args)
        print('Submitting...')
        completed = subprocess.run(['sbatch', str(batch_path.resolve())])
    else:
        instance.run()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('base_dir', type=str)
    parser.add_argument('variant', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--careful', action='store_true')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--partition', type=str)
    parser.add_argument('--nodelist', type=str)
    launch(parser.parse_args())

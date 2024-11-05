#!/usr/bin/env python

from datetime import datetime
import itertools
from pathlib import Path
import random

from force.defaults import DATETIME_FORMAT
from force.util import random_string, try_parse, yes_no_prompt
from force.workflow import slurm


def launch(additional_sbatch_args, nodes, main, configs, sets):
    base_cmd = f'python {main}'
    working_dir = Path.cwd()

    sbatch_args = slurm.SBATCH_DEFAULTS.copy()
    sbatch_args.update(additional_sbatch_args)

    processed_args = {}
    variant_keys = []
    for key, value in sets.items():
        if value.startswith('@'):
            variant_keys.append(key)
            processed_args[key] = [try_parse(s) for s in value[1:].split(',')]
        else:
            processed_args[key] = try_parse(value)

    # Make sure it all looks ok
    print('Processed args:')
    print('\tsbatch:')
    for key, values in sbatch_args.items():
        print(f'\t\t{key}: {values}')
    print('\tConfigs:')
    for c in configs:
        print(f'\t\t{c}')
    print('\tScript:')
    for key, values in processed_args.items():
        print(f'\t\t{key}: {values}')

    if not yes_no_prompt('Accept?'):
        print('Exiting')
        exit()

    # For generating run dir names
    now_str = datetime.now().strftime(DATETIME_FORMAT)
    random.seed()

    # Build the variant map, which maps a variant name to the corresponding command
    variant_prod = itertools.product(*[processed_args[key] for key in variant_keys])
    variant_map = {}
    for variant in variant_prod:
        # variant_name = str(name)
        variant_args = {}
        for key, value in zip(variant_keys, variant):
            if value == '__action__':
                # variant_name += f'_{key}'
                variant_args[key] = ''
            else:
                # variant_name += f'_{key}={value}'
                variant_args[key] = value
        all_args = processed_args.copy()
        all_args.update(variant_args)
        # all_args['slurm-batchfile'] = variant_name
        for key, value in all_args.items():
            if value == '__action__':
                all_args[key] = ''

        rand_str = random_string(4, include_uppercase=False, include_digits=False)
        run_id = f'{now_str}_{rand_str}'
        final_cmd = base_cmd + f' --run-id {run_id} ' + \
                    ' '.join([f'-c {c}' for c in configs]) + ' ' + \
                    ' '.join([f'-s {k} {v}' for k, v in all_args.items()])
        # variant_map[variant_name] = final_cmd
        variant_map[run_id] = final_cmd

    for i, (name, command) in enumerate(variant_map.items()):
        if nodes is None:
            variant_sbatch_args = sbatch_args
        else:
            assigned_node = nodes[i % len(nodes)]
            variant_sbatch_args = dict(sbatch_args, nodelist=assigned_node)
        print(f'Writing {name}')
        slurm.write_batchfile(variant_sbatch_args,
                              conda_env='force', working_dir=working_dir, name=name, command=command)

    jobs = {}
    for name in variant_map:
        job_id = slurm.submit(name)
        if job_id is not None:
            jobs[job_id] = name

    configs_cat = ','.join(configs)
    slurm.new_group(configs_cat, jobs)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('main')
    parser.add_argument('-c', '--config', action='append', default=[])
    parser.add_argument('-s', '--set', nargs=2, action='append', default=[])
    parser.add_argument('--sbatch', nargs=2, action='append', default=[])
    parser.add_argument('--partition', required=True)
    parser.add_argument('--account', required=True)
    parser.add_argument('--nodes', default=None)
    parser.add_argument('--gpu', action='store_true') # shorthand for --sbatch gres gpu:1
    args = parser.parse_args()
    sbatch_args = dict(args.sbatch)
    sbatch_args['partition'] = args.partition
    sbatch_args['account'] = args.account
    nodes = None if args.nodes is None else args.nodes.split(',')
    if args.gpu:
        sbatch_args['gres'] = 'gpu:1'
    elif not yes_no_prompt('GPU will not be used. OK?'):
        print('Exiting')
        exit()
    launch(sbatch_args, nodes, args.main, args.config, dict(args.set))
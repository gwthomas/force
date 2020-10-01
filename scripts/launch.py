#!/usr/bin/env python

import itertools
from pathlib import Path

from force.workflow import cli, slurm


def try_parse(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        for type in (int, float):
            try:
                return type(s)
            except:
                pass
        return s

def launch(additional_sbatch_args, main, configs, sets):
    name = str(main)
    base_cmd = f'python {main}'
    working_dir = Path.cwd()

    sbatch_args = slurm.SBATCH_DEFAULTS.copy()
    for key, value in additional_sbatch_args:
        sbatch_args[key] = value

    processed_args = {}
    variant_keys = []
    for key, value in sets.items():
        if key.startswith('@'):
            base_key = key[1:]
            variant_keys.append(base_key)
            processed_args[base_key] = [try_parse(s) for s in value.split(',')]
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

    if not cli.yes_no_prompt('Accept?'):
        print('Exiting')
        exit()

    # Build the variant map, which maps a variant name to the corresponding command
    variant_prod = itertools.product(*[processed_args[key] for key in variant_keys])
    variant_map = {}
    for variant in variant_prod:
        variant_name = str(name)
        variant_args = {}
        for key, value in zip(variant_keys, variant):
            if value == '__action__':
                variant_name += f'_{key}'
                variant_args[key] = ''
            else:
                variant_name += f'_{key}={value}'
                variant_args[key] = value
        all_args = processed_args.copy()
        all_args.update(variant_args)
        # all_args['slurm-batchfile'] = variant_name
        for key, value in all_args.items():
            if value == '__action__':
                all_args[key] = ''

        final_cmd = base_cmd + ' ' + \
                    ' '.join([f'-c {c}' for c in configs]) + ' ' + \
                    ' '.join([f'-s {k} {v}' for k, v in all_args.items()])
        variant_map[variant_name] = final_cmd

    for name, command in variant_map.items():
        print('Writing', name)
        slurm.write_batchfile(sbatch_args, conda_env='force', working_dir=working_dir, name=name, command=command)

    jobs = {}
    for name in variant_map:
        job_id = slurm.submit1(name)
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
    parser.add_argument('--nodelist', default=None) # shorthand for --sbatch nodelist ____
    parser.add_argument('--gpu', action='store_true') # shorthand for --sbatch gres gpu:1
    args = parser.parse_args()
    sbatch_args = list(args.sbatch)
    if args.nodelist is not None:
        sbatch_args.append(['nodelist', args.nodelist])
    if args.gpu:
        sbatch_args.append(['gres', 'gpu:1'])
    launch(sbatch_args, args.main, args.config, dict(args.set))
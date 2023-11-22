from datetime import datetime

from pathlib import Path
import pickle
from random import randint
import subprocess

import pandas as pd

from force.util import yes_no_prompt, cd


SBATCH_DEFAULTS = {
    'ntasks': 1,
    'cpus-per-task': 1,
    'time': '3-0', # days-hours
    'mem-per-cpu': '4G'
}

SLURM_DIR = Path('~/.slurm').expanduser()
SLURM_DIR.mkdir(parents=True, exist_ok=True)
METADATA_PATH = SLURM_DIR / 'meta.pkl'

ID_MAX = 10**4


def parse_squeue():
    squeue_out = subprocess.run(['squeue', '-u', Path.home().name, '-o', '%.10i %.25j %.2t %.10M %.6D %R'],
                                stdout=subprocess.PIPE, encoding='utf-8').stdout
    split = [line.split() for line in squeue_out.splitlines()]
    columns = split[0]
    data = split[1:]
    expected_types = [int, str, str, str, int, str]
    data = [[t(d) for t, d in zip(expected_types, row)] for row in data]
    return pd.DataFrame(columns=columns, data=data)


def get_metadata():
    if METADATA_PATH.is_file():
        with METADATA_PATH.open('rb') as f:
            return pickle.load(f)
    else:
        return {}

def write_metadata(metadata):
    with METADATA_PATH.open('wb') as f:
        pickle.dump(metadata, f)


def update_metadata():
    squeue_info = parse_squeue()
    active_job_ids = set(squeue_info.JOBID)
    metadata = get_metadata()
    for group_id, group_data in metadata.items():
        for job_id in list(group_data['active jobs'].keys()):
            if job_id not in active_job_ids:
                print('Job', job_id, 'is no longer active')
                group_data['completed jobs'][job_id] = group_data['active jobs'].pop(job_id)
    write_metadata(metadata)


def clear():
    if not yes_no_prompt('The metadata will also be cleared. Proceed?'):
        return

    # Cancel all of the user's currently running jobs
    username = Path.home().name
    subprocess.run(['scancel', '-u', username])

    # Delete all slurm-related files (including the metadata!)
    for slurm_file in SLURM_DIR.iterdir():
        slurm_file.unlink()


def _print_jobs(description, jobs):
    print(f'\t{description}:')
    for id, name in jobs.items():
        print(f'\t\t{id}: {name}')

def print_metadata():
    metadata = get_metadata()
    if not metadata:
        print('Metadata not found!')
        return

    total_active_jobs, total_completed_jobs = 0, 0
    ids_by_time = [k for k, v in sorted(metadata.items(), key=lambda item: item[1]['timestamp'])]
    for id in ids_by_time:
        entry = metadata[id]
        print(f'Group {id}:')
        print('\tTime:', entry['timestamp'])
        print('\tConfig:', entry['config'])
        _print_jobs('Active jobs', entry['active jobs'])
        _print_jobs('Completed jobs', entry['completed jobs'])
        total_active_jobs += len(entry['active jobs'])
        total_completed_jobs += len(entry['completed jobs'])
    print(total_active_jobs, 'active jobs')
    print(total_completed_jobs, 'completed jobs')


def write_batchfile(sbatch_args, conda_env, working_dir, name, command):
    sbatch_args = sbatch_args.copy()
    sbatch_args['job-name'] = name.replace('=', ':')

    lines = ['#!/bin/bash'] + \
            [f'#SBATCH --{key}={value}' for key, value in sbatch_args.items()] + \
            ['source ~/.bashrc',
             f'conda activate {conda_env}',
             f'cd {working_dir}',
             command]
    text = '\n'.join(lines)
    (SLURM_DIR/f'{name}.sh').write_text(text)


def new_group(config, jobs):
    new_group_id = randint(0, ID_MAX)
    metadata = get_metadata()
    while new_group_id in metadata:
        new_group_id = randint(0, ID_MAX)
    metadata[new_group_id] = {
        'config': config,
        'timestamp': datetime.now(),
        'active jobs': jobs,
        'completed jobs': {}
    }
    write_metadata(metadata)

    n_jobs = len(jobs)
    job_ids = sorted(list(jobs.keys()))
    print(f'Created new group {new_group_id} with {n_jobs} jobs: {job_ids}')


def submit(name):
    print('Submitting', name)
    with cd(SLURM_DIR):
        proc = subprocess.run(['sbatch', '--requeue', f'{name}.sh'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              encoding='utf-8')
        print(proc.stdout)
        if proc.stdout.startswith('Submitted'):
            job_id = int(proc.stdout.strip().split(' ')[-1])
            return job_id
        else:
            print('Failed to submit! stderr:')
            print(proc.stderr)
            exit()


def cancel(group_id):
    metadata = get_metadata()
    if group_id not in metadata:
        print(f'Item {group_id} does not appear in our records. It must not exist.')
        print('Existing batch jobs:', sorted(list(metadata.keys())))
        return

    active_jobs = metadata[group_id]['active jobs']
    print('Canceling the following Slurm batch jobs:', list(active_jobs.keys()))
    for job_id in active_jobs:
        subprocess.run(['scancel', str(job_id)])

    del metadata[group_id]
    write_metadata(metadata)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('-p', '--print', action='store_true')
    parser.add_argument('-c', '--cancel', type=int, default=None)
    args = parser.parse_args()

    if args.clear:
        clear()

    if args.print:
        update_metadata()
        print_metadata()

    if args.cancel is not None:
        cancel(args.cancel)
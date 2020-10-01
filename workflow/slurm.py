from contextlib import contextmanager
from datetime import datetime
import os

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

import itertools
import json
from pathlib import Path
import pdb
import pickle
from random import randint
import readline
import subprocess

import pandas as pd

from force import MACHINE_VARIABLES
from .cli import yes_no_prompt


def parse_squeue():
    squeue_out = subprocess.run(['squeue', '-u', Path.home().name],
                                stdout=subprocess.PIPE, encoding='utf-8').stdout
    split = [line.split() for line in squeue_out.splitlines()]
    columns = split[0]
    data = split[1:]
    expected_types = [int, str, str, str, str, str, int, str]
    data = [[t(d) for t, d in zip(expected_types, row)] for row in data]
    return pd.DataFrame(columns=columns, data=data)


SBATCH_DEFAULTS = {
    'ntasks': 1,
    'partition': MACHINE_VARIABLES['default-partition'],
    'cpus-per-task': 1,
    'time': '2-0',
    'mem-per-cpu': '4G'
}

SLURM_DIR = Path(MACHINE_VARIABLES['slurm-dir'])
SLURM_DIR.mkdir(parents=True, exist_ok=True)
METADATA_PATH = SLURM_DIR / 'meta.pkl'

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
    subprocess.run(['scancel', '-u', Path.home().name])

    for slurm_file in SLURM_DIR.iterdir(): # This includes the metadata
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
    ID_MAX = int(1e4)
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

def submit1(name):
    print('Submitting', name)
    with cd(SLURM_DIR):
        proc = subprocess.run(['sbatch', f'{name}.sh'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              encoding='utf-8')
        print(proc.stdout)
        if proc.stdout.startswith('Submitted'):
            job_id = int(proc.stdout.strip().split(' ')[-1])
            return job_id
        else:
            print('Failed to submit', name)
            return None

def launch(config_path, process_config, override, additional_sbatch_args):
    config_path = Path(config_path)
    assert config_path.is_file()
    sbatch_args, conda_env, working_dir, variant_map = process_config(
        json.loads(config_path.read_text()),
        override,
        additional_sbatch_args
    )

    for name, command in variant_map.items():
        print('Writing', name)
        write_batchfile(sbatch_args, conda_env, working_dir, name, command)
        write_batchfile(sbatch_args, conda_env, working_dir, f'{name}_resume', f'{command} --resume')

    jobs = {}
    for name in variant_map:
        job_id = submit1(name)
        if job_id is not None:
            jobs[job_id] = name
    new_group(config_path.name, jobs)

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

def resubmit(name):
    metadata = get_metadata()
    for group_id, group_data in metadata.items():
        all_jobs = {**group_data['active jobs'], **group_data['completed jobs']}
        for job_id, job_name in all_jobs.items():
            if name == job_id or name == job_name:
                # Found the job in the metadata, so we know which group it came from. Now relaunch it
                new_job_id = submit1(f'{name}_resume')
                if new_job_id is not None:
                    group_data['active jobs'][new_job_id] = name
                    print(f'Successfully resubmitted {name} under new job ID {new_job_id}')
                    write_metadata(metadata)
                else:
                    print('Failed to resubmit')

                # Exit early so that we don't resubmit several times. If it's already been resubmitted, there will be
                # more than one entry with the given name in the metadata
                return
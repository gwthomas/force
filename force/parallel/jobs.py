from multiprocessing.synchronize import Lock
from pathlib import Path
from subprocess import Popen, PIPE

import torch.multiprocessing as mp

from force.config import Configurable
from force.log import Log
from force.util import queue_safe_get


type JobInfo = tuple[str, str, Popen]


POLL_PERIOD = 1.0


def job_daemon_main(cmdq: mp.Queue, log_path: Path, log_lock: Lock):
    log = Log(log_path, log_lock)
    active_jobs: list[JobInfo] = []
    while True:
        prejob_info = queue_safe_get(cmdq, timeout=POLL_PERIOD)
        if prejob_info is not None:
            name, cmd = prejob_info
            proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
            active_jobs.append((name, cmd, proc))
        
        # Check if jobs have terminated
        next_active_jobs: list[JobInfo] = []
        for job_info in active_jobs:
            name, cmd, proc = job_info
            if proc.poll() is None:
                # Job still running
                next_active_jobs.append(job_info)
            else:
                # Job has completed
                stdout, stderr = map(lambda bs: str(bs, encoding='utf-8'), proc.communicate())
                log(f'Completed asynchronous job {name}. Output:\n{stdout}')
                if stderr:
                    raise RuntimeError(stderr)
        active_jobs = next_active_jobs


class JobManager:
    def __init__(self, log: Log):
        self._cmdq = mp.Queue()

        args = (self._cmdq, log.path, log.lock)
        self._daemon = mp.Process(target=job_daemon_main, args=args, daemon=True)
        self._daemon.start()

    def enqueue(self, name: str, cmd: str):
        self._cmdq.put((name, cmd))
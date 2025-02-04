from datetime import datetime
from os import PathLike
from pathlib import Path

import torch.multiprocessing as mp


class Log:
    def __init__(self, path=None, lock=None):
        if path is None:
            self._path = None
            self._file = None
        else:
            self._path = Path(path)
            if self._path.exists():
                print(f'WARNING: overwriting existing log at {self.path}')
            self._file = self._path.open('w', buffering=1)

        self._lock = mp.Lock() if lock is None else lock

    @property
    def path(self):
        return self._path
    
    @property
    def lock(self):
        return self._lock

    def write(self, message: str, timestamp=True):
        if timestamp:
            now_str = datetime.now().strftime('%H:%M:%S')
            message = f'[{now_str}] ' + message
        else:
            message = ' ' * 11 + message

        with self._lock:
            print(message, flush=True)
            if self._file is not None:
                self._file.write(f'{message}\n')
                self._file.flush()

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def close(self):
        if self._file is not None:
            with self._lock:
                self._file.close()
            self._file = None


__global_log = None
def set_global_log(log: Log):
    global __global_log
    __global_log = log

def get_global_log():
    return __global_log
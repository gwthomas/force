from pathlib import Path


class Logger:
    def __init__(self):
        self._dir = None
        self._log_file = None

    def setup(self, dir, log_filename='log.txt', mode='w'):
        assert self._dir is None, 'Can only setup once'
        self._dir = Path(dir)
        self._dir.mkdir(exist_ok=True)
        print(f'Set log dir to {dir}, log filename to {log_filename}')
        self._log_file = (self._dir / log_filename).open(mode, buffering=1)

    @property
    def dir(self):
        return self._dir

    def message(self, message):
        print(message)
        self._log_file.write(f'{message}\n')


default_logger = Logger()
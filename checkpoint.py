from copy import deepcopy
from pathlib import Path
import torch


class CheckpointableData:
    def __init__(self):
        self._data = {}

    def __getitem__(self, item):
        return self._data[item]

    def append(self, name, value, verbose=False):
        if name not in self._data:
            self._data[name] = []
        self._data[name].append(value)
        if verbose:
            from .log import default_logger as log
            log.message(f'{name}: {value}')

    def state_dict(self):
        return deepcopy(self._data)

    def load_state_dict(self, state_dict):
        self._data = deepcopy(state_dict)

    def __repr__(self):
        return repr(self._data)


def assert_checkpointable(o):
    """An object is considered checkpointable if it
        (1) has both the state_dict and load_state_dict methods, or
        (2) is a list containing only checkpointable objects
        (3) is a dict containing only checkpointable objects as values (the keys may be whatever)
    """
    if isinstance(o, list):
        for item in o:
            assert_checkpointable(item)
    elif isinstance(o, dict):
        for item in o.values():
            assert_checkpointable(item)
    else:
        assert hasattr(o, 'state_dict')
        assert callable(o.state_dict)
        assert hasattr(o, 'load_state_dict')
        assert callable(o.load_state_dict)


class Checkpointer:
    def __init__(self, checkpointables, dir, filename_format):
        assert_checkpointable(checkpointables)
        self.checkpointables = checkpointables
        self.dir = Path(dir)
        self.filename_format = filename_format

    def _path(self, *args):
        return self.dir / self.filename_format.format(*args)

    def save(self, *args):
        if isinstance(self.checkpointables, list):
            state = [checkpointable.state_dict() for checkpointable in self.checkpointables]
        elif isinstance(self.checkpointables, dict):
            state = {name: checkpointable.state_dict() for name, checkpointable in self.checkpointables.items()}
        else:
            state = self.checkpointables.state_dict()
        torch.save(state, self._path(*args))

    def load(self, *args):
        loaded_state = torch.load(self._path(*args), map_location='cpu')
        if isinstance(self.checkpointables, list):
            assert isinstance(loaded_state, list)
            for checkpointable, state_dict in zip(self.checkpointables, loaded_state):
                checkpointable.load_state_dict(state_dict)
        elif isinstance(self.checkpointables, dict):
            assert isinstance(loaded_state, dict)
            for name, checkpointable in self.checkpointables.items():
                checkpointable.load_state_dict(loaded_state[name])
        else:
            self.checkpointables.load_state_dict(loaded_state)

    def try_load(self, *args):
        try:
            self.load(*args)
            return True
        except:
            return False

    def load_latest(self, candidates):
        for candidate in sorted(candidates, reverse=True):
            if self.try_load(candidate):
                return candidate
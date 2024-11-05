from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import random
import string

import numpy as np
import torch

from force import defaults


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def pymean(x):
    return sum(x) / len(x)


def try_parse(s):
    try:
        return eval(s)
    except:
        return s

def yes_no_prompt(prompt):
    response = input(f'{prompt} y/n ')
    if response in {'y', 'yes'}:
        return True
    elif response in {'n', 'no'}:
        return False
    else:
        raise RuntimeError('Invalid response to prompt (expected y/yes or n/no)')


def random_string(n, include_lowercase=True, include_uppercase=False, include_digits=False):
    alphabet = ''
    if include_lowercase:
        alphabet += string.ascii_lowercase
    if include_uppercase:
        alphabet += string.ascii_uppercase
    if include_digits:
        alphabet += string.digits
    return ''.join(random.choices(alphabet, k=n))


def load_configd(path):
    if isinstance(path, str):
        path = Path(path)
    else:
        assert isinstance(path, Path)
    assert path.is_file(), f'No file at path {path}'

    import yaml
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    with path.open('r') as f:
        cfgd = yaml.load(f, Loader=Loader)

    assert isinstance(cfgd, dict), 'Config yaml should parse to a dict'
    return cfgd


def update_cfgd(dst: dict, src: dict):
    assert isinstance(dst, dict)
    assert isinstance(src, dict)
    for k, v in src.items():
        if isinstance(v, (bool, int, float, str)):
            dst[k] = v
        elif isinstance(v, dict):
            if k not in dst:
                dst[k] = {}
            update_cfgd(dst[k], v)
        else:
            raise ValueError(f'Invalid object in config: {v}')


def unique_value(name):
    """Creates a unique value with the given name.
    The value has its own class, of which it is the only instance."""
    class UniqueClass:
        def __str__(self):
            return name
        def __repr__(self):
            return name
    UniqueClass.__name__ = name
    return UniqueClass()


def dict_get(dict, *args):
    return (dict[key] for key in args)

def prefix_dict_keys(prefix: str, new: dict):
    return {f'{prefix}/{k}': v for k, v in new.items()}


def discounted_sum(rewards, discount):
    exp_discounts = discount**torch.arange(len(rewards), dtype=torch.float, device=rewards.device)
    return torch.sum(rewards * exp_discounts)


def time_since(start_t: datetime):
    end_t = datetime.now()
    return end_t - start_t


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def read_tfevents(path, tags):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if isinstance(tags, str):
        single = True
        tags = [tags]
    else:
        single = False
        assert type(tags) in {list, tuple}

    path = str(path)
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    all_results = {}
    for tag in tags:
        results = []
        try:
            for e in event_acc.Scalars(tag):
                results.append((e.step, e.value))
        except KeyError:
            pass
        all_results[tag] = results
    if single:
        return results
    else:
        return all_results


def compute_returns(rewards, discount):
    T = len(rewards)
    returns = torch.zeros_like(rewards)
    returns[-1] = rewards[-1]
    for t in reversed(range(T-1)):
        returns[t] = rewards[t] + discount * returns[t+1]
    return returns
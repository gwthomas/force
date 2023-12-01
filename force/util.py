from contextlib import contextmanager
from datetime import datetime
from math import ceil
import os
import random
import string

import numpy as np
import torch


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


def dict_get_several(dict, *args):
    return (dict[key] for key in args)

def prefix_dict_keys(prefix: str, new: dict):
    return {f'{prefix}/{k}': v for k, v in new.items()}


def discounted_sum(rewards, discount):
    exp_discounts = discount**torch.arange(len(rewards), dtype=torch.float, device=rewards.device)
    return torch.sum(rewards * exp_discounts)


def batch_iterator(args, batch_size=1000, shuffle=False):
    if type(args) in {list, tuple}:
        multi_arg = True
        n = len(args[0])
        for i, arg_i in enumerate(args):
            assert isinstance(arg_i, torch.Tensor)
            assert len(arg_i) == n
    else:
        multi_arg = False
        n = len(args)

    indices = torch.randperm(n) if shuffle else torch.arange(n)
    n_batches = ceil(float(n) / batch_size)
    for batch_index in range(n_batches):
        batch_start = batch_size * batch_index
        batch_end = min(batch_size * (batch_index + 1), n)
        batch_indices = indices[batch_start:batch_end]
        if multi_arg:
            yield [arg[batch_indices] for arg in args]
        else:
            yield args[batch_indices]


def batch_map(fn, args, batch_size=1000):
    if type(args) in {list, tuple}:
        results = [fn(*batch) for batch in batch_iterator(args, batch_size=batch_size)]
    else:
        results = [fn(batch) for batch in batch_iterator(args, batch_size=batch_size)]

    proto = results[0]
    if isinstance(proto, torch.Tensor):
        return torch.cat(results)
    elif isinstance(proto, tuple):
        assert all(isinstance(x, torch.Tensor) for x in proto)
        n = len(proto)
        return tuple(torch.cat([x[i] for x in results]) for i in range(n))
    elif isinstance(proto, dict):
        assert all(isinstance(x, torch.Tensor) for x in proto.values())
        keys = list(proto.keys())
        return {k: torch.cat([d[k] for d in results]) for k in keys}
    else:
        raise ValueError('batch_map can only by applied to functions which outputs tensors or tuples/dicts of tensors')


def time_since(start_t):
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


from math import ceil
import random

import numpy as np
import torch
from tqdm import trange

from force.torch_util import device, torchify

def set_seed(seed):
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def discounted_sum(rewards, discount):
    return torch.sum(rewards * discount**torch.arange(len(rewards), dtype=torch.float, device=device))

def batch_map(fn, args, batch_size=256, progress_bar=False):
    if type(args) in (list, tuple):
        n = len(args[0])
        for i, arg_i in enumerate(args):
            assert isinstance(arg_i, torch.Tensor)
            assert len(arg_i) == n
    else:
        n = len(args)
        args = [args]

    n_batches = ceil(float(n) / batch_size)
    iter_range_fn = trange if progress_bar else range
    results = []
    for batch_index in iter_range_fn(n_batches):
        batch_start = batch_size * batch_index
        batch_end = min(batch_size * (batch_index + 1), n)
        batch_output = fn(*[arg[batch_start:batch_end] for arg in args])
        results.append(batch_output)
    return torch.cat(results)
import datetime

import torch

from gtml.constants import DEFAULT_TIMESTAMP_FORMAT


def scalar(x):
    if type(x) in (int, float):
        return x
    elif isinstance(x, torch.Tensor):
        return x.item()
    else:
        raise ValueError('Could not convert {} to scalar'.format(x))

def count_parameters(model, trainable_only=True):
    params = model.parameters()
    if trainable_only:
        params = [p for p in params if p.requires_grad]
    return sum(p.numel() for p in params)
        
def one_hot(labels):
    n = len(torch.unique(labels))
    return torch.eye(n)[labels]


def timestamp(format_string=DEFAULT_TIMESTAMP_FORMAT):
    now = datetime.datetime.now()
    return now.strftime(format_string)


def set_random_seed(seed):
    torch.manual_seed(seed)
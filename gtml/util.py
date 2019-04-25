import datetime

import torch

from gtml.constants import DEFAULT_TIMESTAMP_FORMAT


def one_hot(labels):
    n = len(torch.unique(labels))
    return torch.eye(n)[labels]


def timestamp(format_string=DEFAULT_TIMESTAMP_FORMAT):
    now = datetime.datetime.now()
    return now.strftime(format_string)


def set_random_seed(seed):
    torch.manual_seed(seed)
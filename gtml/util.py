import datetime

import numpy as np
import torch

from gtml.constants import DEFAULT_TIMESTAMP_FORMAT, DEVICE


def one_hot(labels):
    n = len(torch.unique(labels))
    return torch.eye(n)[labels]

def parameter_vector(model, trainable_only=True):
    params = model.parameters()
    if trainable_only:
        params = [p for p in params if p.requires_grad]
    return torch.cat([p.view(-1) for p in params])

def transfer_recursive(x):
    """Given a tensor or a container which contains tensors, returns the same
       structure but with all tensors moved to DEVICE.
       Any non-tensor leaf objects are left as-is."""
    if isinstance(x, torch.Tensor):
        return x.to(DEVICE)
    elif isinstance(x, list):
        return [transfer_recursive(elem) for elem in x]
    elif isinstance(x, tuple):
        return tuple(transfer_recursive(elem) for elem in x)
    elif isinstance(x, dict):
        return {key: transfer_recursive(value) for key, value in dict.items()}
    elif isinstance(x, set):
        return {transfer_recursive(elem) for elem in x}
    else:
        return x

def timestamp(format_string=DEFAULT_TIMESTAMP_FORMAT):
    now = datetime.datetime.now()
    return now.strftime(format_string)

def set_random_seed(seed, env=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if env is not None:
        env.seed(seed)

import numpy as np
import torch
import torch.nn as nn

from force import defaults


# Any activation functions added below can be looked up by name
NAMED_ACTIVATIONS = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'softmax': lambda: nn.Softmax(dim=-1),
    'softplus': nn.Softplus,
    'tanh': nn.Tanh
}


TORCH_INT_TYPES = {
    torch.int8, torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64
}


def get_initializer(name: str):
    return getattr(nn.init, f'{name}_')


def get_device(d):
    if d is None:
        return defaults.DEVICE
    elif isinstance(d, torch.device):
        return d
    elif isinstance(d, str):
        return torch.device(d)
    else:
        raise ValueError(f'Invalid device: {d}')


def torchify(x, double_to_float=True, int_to_long=True, device=None):
    device = get_device(device)

    # Recursive cases
    if type(x) in {list, tuple}:
        return type(x)(torchify(xx, double_to_float, int_to_long, device) for xx in x)
    if isinstance(x, dict):
        return {k: torchify(v, double_to_float, int_to_long, device) for k, v in x.items()}

    if torch.is_tensor(x):
        pass
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        x = torch.tensor(x)

    if x.dtype == torch.double:
        if double_to_float:
            x = x.float()
    elif x.dtype in TORCH_INT_TYPES:
        if int_to_long:
            x = x.long()

    return x.to(device)


def numpyify(x):
    if isinstance(x, np.ndarray):
        return x
    elif torch.is_tensor(x):
        return x.cpu().numpy()
    else:
        return np.array(x)


# Wrapper to hide modules from parent modules
class ModuleWrapper:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, attr):
        return getattr(self.module, attr)


def random_indices(high, size=None, replace=False, p=None):
    assert isinstance(high, int)
    p_np = numpyify(p) if torch.is_tensor(p) else p
    return torchify(np.random.choice(high, size=size, replace=replace, p=p_np))

def random_choice(tensor, size=None, replace=False, p=None, dim=0):
    indices = random_indices(tensor.shape[dim], size=size, replace=replace, p=p)
    return tensor.index_select(dim, indices)

def quantile(a, q, dim=None, as_torch=True, device='cpu'):
    quantile = np.quantile(numpyify(a), numpyify(q), axis=dim)
    if as_torch:
        return torchify(quantile, device=device)
    elif quantile.ndim == 0:
        return float(quantile)
    else:
        return list(map(float, quantile))

def quartiles(a, include0=True, include1=True, **kwargs):
    q = ([0.0] if include0 else []) + [0.25, 0.5, 0.75] + ([1.0] if include1 else [])
    return quantile(a, q, **kwargs)

def deciles(a, include0=True, include1=True, **kwargs):
    q = ([0.0] if include0 else []) + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + ([1.0] if include1 else [])
    return quantile(a, q, **kwargs)


def update_ema(target, source, rate):
    assert 0 <= rate <= 1
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(rate * param.data + (1 - rate) * target_param.data)


def select1_per_row(matrix, indices):
    assert matrix.ndim == 2
    assert indices.ndim == 1 and indices.dtype is torch.int64
    assert matrix.shape[0] == len(indices)
    return matrix.gather(1, indices.unsqueeze(1)).squeeze(1)


def freepeat(x, repetitions, dim):
    """Repeats a tensor along a new axis.
    It is "free" because expand does not copy data.
    """
    expand_arg = [-1 for _ in range(x.ndim+1)]
    expand_arg[dim] = repetitions
    return torch.unsqueeze(x, dim=dim).expand(*expand_arg)


def _mem_str(m):
    if m > 10**9:
        return f'{m / 10**9:.1f} GB'
    elif m > 10**6:
        return f'{m // 10**6:.1f} MB'
    elif m > 1000:
        return f'{m // 10**3:.1f} KB'
    else:
        return f'{m} B'

def gpu_mem_info(as_str=True):
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    info = {
        'total': t,
        'reserved': r,
        'allocated': a,
        'reserved but unallocated': r - a
    }
    if as_str:
        return {k: _mem_str(v) for k, v in info.items()}
    else:
        return info
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

    # If we got here, x should be a single tensor-like object
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


def random_indices(high, size=None, replace=False, p=None, device=None):
    assert isinstance(high, int)
    p_np = numpyify(p) if torch.is_tensor(p) else p
    return torchify(np.random.choice(high, size=size, replace=replace, p=p_np), device=device)

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


def batch_iterator(args, batch_size, shuffle=False, device=None):
    if type(args) in {list, tuple}:
        which = 'sequence'
        n = len(args[0])
        for i, arg_i in enumerate(args):
            assert isinstance(arg_i, torch.Tensor)
            assert len(arg_i) == n
    elif isinstance(args, dict):
        which = 'dict'
        values = tuple(args.values())
        n = len(values[0])
        for v in values:
            assert isinstance(v, torch.Tensor)
            assert len(v) == n
    else:
        which = 'single'
        n = len(args)

    indices = torch.randperm(n) if shuffle else torch.arange(n)
    batch_start = 0
    while batch_start < n:
        batch_end = min(batch_start + batch_size, n)
        batch_indices = indices[batch_start:batch_end]
        if which == 'sequence':
            result = [arg[batch_indices] for arg in args]
        elif which == 'dict':
            result = {k: v[batch_indices] for k, v in args.items()}
        else:
            result = args[batch_indices]

        if device is not None:
            result = torchify(result, device=device)

        yield result

        batch_start = batch_end

def batch_map(fn, args, batch_size=defaults.BATCH_MAP_SIZE, batch_device=None):
    """Used to compute fn(args) (or fn(*args)) where the args tensor(s) may be
    large enough to cause an out-of-memory error if evaluated all at once.
    This function breaks the args up into batches, applies fn to each batch,
    and concatenates the results back together.
    """
    iterator_args = (args, batch_size, False, batch_device)
    if type(args) in {list, tuple}:
        results = [fn(*batch) for batch in batch_iterator(*iterator_args)]
    else:
        results = [fn(batch) for batch in batch_iterator(*iterator_args)]

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



def collapse_map(f, x):
    """Applies f, a function which maps [b, d] -> [b, d'], where
        b is an arbitrary batch dimension,
        d is the input feature dimension,
        d' is the output feature dimension
    to x, a tensor of shape [b1, ..., bk, d], to produce an output of shape
    [b1, ..., bk, d'].
    The batch dimensions b1, ..., bk will be collapsed to a single dimension b,
    f applied, then b reshaped to b1, ..., bk.
    """
    bs = x.shape[:-1]
    b = torch.tensor(bs).prod()
    x = x.reshape(b, -1)
    fx = f(x)
    assert len(fx) == b, f'collapse_map f must maintain the batch dimension'
    if fx.ndim == 1:
        return fx.reshape(*bs)
    elif fx.ndim == 2:
        return fx.reshape(*bs, -1)
    else:
        raise ValueError('Too many output dimensions for collapse_map')


def keywise_stack(list_of_dicts, dim=0):
    keys = tuple(list_of_dicts[0].keys())
    ret = {k: [] for k in keys}
    for d in list_of_dicts:
        for k in keys:
            ret[k].append(d[k])
    return {k: torch.stack(ret[k], dim=dim) for k in keys}


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
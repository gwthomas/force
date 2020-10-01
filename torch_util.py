import operator
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def torchify(x, double_to_float=True, int_to_long=True, to_device=True):
    if torch.is_tensor(x):
        pass
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        x = torch.tensor(x)

    if x.dtype == torch.double:
        if double_to_float:
            x = x.float()
    elif x.dtype == torch.int:
        if int_to_long:
            x = x.long()

    if to_device:
        x = x.to(device)

    return x

def numpyify(x):
    if isinstance(x, np.ndarray):
        return x
    elif torch.is_tensor(x):
        return x.cpu().numpy()
    else:
        return np.array(x)


def random_indices(high, size=None, replace=False, p=None):
    assert isinstance(high, int)
    p_np = numpyify(p) if torch.is_tensor(p) else p
    return torchify(np.random.choice(high, size=size, replace=replace, p=p_np))

def random_choice(tensor, size=None, replace=False, p=None, dim=0):
    indices = random_indices(tensor.shape[dim], size=size, replace=replace, p=p)
    return tensor.index_select(dim, indices)

def quantile(a, q, dim=None):
    return torchify(np.quantile(numpyify(a), numpyify(q), axis=dim))


# PyTorch doesn't let you take sum/mean/max of a list of tensors (unlike NumPy); you have to stack it first.
# I found myself doing this often enough to write shortcuts
def sequence_sum(tensors, dim=0):
    return torch.sum(torch.stack(tensors, dim=dim), dim=dim)

def sequence_mean(tensors, dim=0):
    return torch.mean(torch.stack(tensors, dim=dim), dim=dim)

def sequence_max(tensors, dim=0, include_indices=False):
    m = torch.max(torch.stack(tensors, dim=dim), dim=dim)
    return m if include_indices else m.values




class Module(nn.Module):
    def __call__(self, *args, **kwargs):
        args = [x.to(device) if isinstance(x, torch.Tensor) else x for x in args]
        kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return super().__call__(*args, **kwargs)

    def save(self, f, prefix='', keep_vars=False):
        state_dict = self.state_dict(prefix=prefix, keep_vars=keep_vars)
        torch.save(f, state_dict)

    def load(self, f, map_location=None, strict=True):
        state_dict = torch.load(f, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)

    def try_load(self, f, **kwargs):
        try:
            self.load(f, **kwargs)
            return True
        except:
            return False


def pvector_binop(op, a, b):
    list_like_types = {list, tuple}
    a_ll = type(a) in list_like_types
    b_ll = type(b) in list_like_types
    if a_ll and b_ll:
        return [op(a1, b1) for a1, b1 in zip(a, b)]
    elif a_ll:
        return [op(a1, b) for a1 in a]
    elif b_ll:
        return [op(a, b1) for b1 in b]
    else:
        raise ValueError('At least one input must be of type list or tuple')

def pvector_add(a, b):
    return pvector_binop(operator.__add__, a, b)

def pvector_sub(a, b):
    return pvector_binop(operator.__sub__, a, b)

def pvector_mul(a, b):
    return pvector_binop(operator.__mul__, a, b)

def pvector_div(a, b):
    return pvector_binop(operator.__truediv__, a, b)

def average_pvectors(pvecs):
    return [sequence_mean(ps) for ps in zip(*pvecs)]


default_init_w = nn.init.xavier_normal_
default_init_b = nn.init.zeros_

def weight_initializer(init_w=default_init_w, init_b=default_init_b):
    def init_fn(m):
        if isinstance(m, nn.Linear):
            init_w(m.weight)
            init_b(m.bias)
    return init_fn


def dry_run(module, input_dim):
    """Just runs the network forward once and ignores errors.
    Seems to fix an uninformative PyTorch/CUDA error I was having, but not sure why."""
    try:
        with torch.no_grad():
            module(torchify(np.zeros((1, input_dim))))
    except:
        pass


def mlp(dims, layer_class=nn.Linear, activation=nn.ReLU(), output_activation=None):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'
    layers = []
    for i in range(n_dims - 2):
        layers.append(layer_class(dims[i], dims[i+1]))
        layers.append(activation)
    layers.append(layer_class(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation)
    net = nn.Sequential(*layers)
    net.apply(weight_initializer())
    net.to(device=device, dtype=torch.float)
    dry_run(net, dims[0])
    return net


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def update_ema(target, source, rate):
    assert 0 <= rate <= 1
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(rate * param.data + (1 - rate) * target_param.data)


def _mem_str(m):
    if m > 10**9:
        return f'{m // 10**9} GB'
    elif m > 10**6:
        return f'{m // 10**6} MB'
    elif m > 1000:
        return f'{m // 10**3} KB'
    else:
        return f'{m} B'

def gpu_mem_info(as_str=True):
    t = torch.cuda.get_device_properties(0).total_memory
    c = torch.cuda.memory_cached(0)
    a = torch.cuda.memory_allocated(0)
    f = c - a  # free inside cache
    info = {
        'total': t,
        'cached': c,
        'allocated': a,
        'free': f
    }
    if as_str:
        return {k: _mem_str(v) for k, v in info.items()}
    else:
        return info
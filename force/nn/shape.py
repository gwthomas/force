from frozendict import frozendict
import torch


# A shape specifier is defined recursively as any of the following:
#   i) a torch.Size object
#  ii) a tuple of shapes
# iii) a frozendict of shapes
# Tuples and frozendicts are used for hashability.


def is_valid_shape(shape):
    if isinstance(shape, torch.Size):
        return True
    elif isinstance(shape, tuple):
        return all(is_valid_shape(x) for x in shape)
    elif isinstance(shape, frozendict):
        return all(is_valid_shape(x) for x in shape.values())
    else:
        return False


def shape2str(shape):
    if isinstance(shape, torch.Size):
        return str(list(shape))
    elif isinstance(shape, tuple):
        return '(' + ', '.join([shape2str(x) for x in shape]) + ')'
    elif isinstance(shape, frozendict):
        return '{' + ', '.join([f'{k}: {shape2str(v)}' for k, v in shape.items()]) + '}'
    else:
        raise ValueError(f'Not a shape: {shape}')


def get_nonbatch_shape(obj, batch_dims):
    if isinstance(obj, torch.Tensor):
        return obj.shape[batch_dims:]
    elif type(obj) in {list, tuple}:
        return tuple(x.shape[batch_dims:] for x in obj)
    elif isinstance(obj, dict):
        return frozendict({k: v.shape[batch_dims:] for k, v in obj.items()})
    else:
        raise ValueError(f'Cannot determine shape of {obj}')


def matches_shape(obj, shape, batch_dims=1):
    if isinstance(obj, torch.Tensor):
        return isinstance(shape, torch.Size) and \
               obj.shape[batch_dims:] == shape
    elif type(obj) in {list, tuple}:
        return isinstance(shape, tuple) and \
               len(obj) == len(shape) and \
               all(matches_shape(*pair, batch_dims) for pair in zip(obj, shape))
    elif isinstance(obj, dict):
        return isinstance(shape, frozendict) and \
               set(obj.keys()) == set(shape.keys()) and \
               all(matches_shape(obj[k], shape[k], batch_dims) for k in obj.keys())
    else:
        return False


def shape_numel(shape):
    if isinstance(shape, torch.Size):
        return shape.numel()
    elif isinstance(shape, tuple):
        return sum(shape_numel(x) for x in shape)
    elif isinstance(shape, frozendict):
        return sum(shape_numel(v) for v in shape.values())
    else:
        raise ValueError(f'Invalid shape specifier: {shape}')
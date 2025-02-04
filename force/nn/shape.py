import torch
from torch import Tensor, Size

type Shape = Size | list[Shape] | dict[str, Shape]


def is_valid_shape(shape):
    if isinstance(shape, Size):
        return True
    elif isinstance(shape, list):
        return all(is_valid_shape(x) for x in shape)
    elif isinstance(shape, dict):
        return all(is_valid_shape(x) for x in shape.values())
    else:
        return False


def shape2str(shape):
    if isinstance(shape, Size):
        return str(list(shape))
    elif isinstance(shape, list):
        return '(' + ', '.join([shape2str(x) for x in shape]) + ')'
    elif isinstance(shape, dict):
        return '{' + ', '.join([f'{k}: {shape2str(v)}' for k, v in shape.items()]) + '}'
    else:
        raise ValueError(f'Not a shape: {shape}')


def get_nonbatch_shape(obj, batch_dims):
    if isinstance(obj, Tensor):
        return obj.shape[batch_dims:]
    elif type(obj) in {list, tuple}:
        return [x.shape[batch_dims:] for x in obj]
    elif isinstance(obj, dict):
        return {k: v.shape[batch_dims:] for k, v in obj.items()}
    else:
        raise ValueError(f'Cannot determine shape of {obj}')


def matches_shape(obj, shape, batch_dims=1):
    if isinstance(obj, Tensor):
        return isinstance(shape, Size) and \
               obj.shape[batch_dims:] == shape
    elif type(obj) in {list, tuple}:
        return isinstance(shape, list) and \
               len(obj) == len(shape) and \
               all(matches_shape(*pair, batch_dims) for pair in zip(obj, shape))
    elif isinstance(obj, dict):
        return isinstance(shape, dict) and \
               set(obj.keys()) == set(shape.keys()) and \
               all(matches_shape(obj[k], shape[k], batch_dims) for k in obj.keys())
    else:
        return False


def shape_numel(shape):
    if isinstance(shape, Size):
        return shape.numel()
    elif isinstance(shape, list):
        return sum(shape_numel(x) for x in shape)
    elif isinstance(shape, dict):
        return sum(shape_numel(v) for v in shape.values())
    else:
        raise ValueError(f'Invalid shape specifier: {shape}')
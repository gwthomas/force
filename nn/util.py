import torch


def batch_flatten(x):
    size = list(x.size())
    return x.view(size[0], torch.IntTensor(size[1:]).prod())

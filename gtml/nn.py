import torch
import torch.nn as nn


class BatchFlatten(nn.Module):
    def forward(self, x):
        shape = list(x.shape)
        return x.view(shape[0], torch.IntTensor(shape[1:]).prod())


def infer_shape(f, in_shape):
    batched_shape = [1] + list(in_shape)
    with torch.set_grad_enabled(False):
        return f(torch.zeros(*batched_shape)).shape[1:]


def mlp(sizes, nl_hidden=nn.ReLU, nl_out=None):
    layers = [nn.Linear(sizes[0], sizes[1])]
    for i in range(1, len(sizes)-1):
        layers.append(nl_hidden())
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
    if nl_out is not None:
        layers.append(n_out())
    return nn.Sequential(*layers)


def atari(in_shape=(4,84,84)):
    relu = nn.ReLU()
    conv = nn.Sequential(
        nn.Conv2d(in_shape[0], 32, 8, stride=4), relu,
        nn.Conv2d(32, 64, 4, stride=2), relu,
        nn.Conv2d(64, 64, 3, stride=1), relu,
        BatchFlatten()
    )
    n_conv_out = infer_shape(conv, in_shape)[0]
    return nn.Sequential(
        conv,
        mlp([n_conv_out, 512])
    )


def lenet(in_shape):
    tanh = nn.Tanh()
    relu = nn.ReLU()
    pool = nn.MaxPool2d(2)
    conv = nn.Sequential(
        nn.Conv2d(in_shape[0], 6, 5), tanh, pool,
        nn.Conv2d(6, 16, 5), tanh, pool,
        nn.Conv2d(16, 120, 3), relu,
        BatchFlatten()
    )
    n_conv_out = infer_shape(conv, in_shape)[0]
    return nn.Sequential(
        conv,
        mlp([n_conv_out, 84, 10])
    )

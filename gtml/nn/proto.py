import torch; nn = torch.nn; F = torch.nn.functional


class BatchFlatten(nn.Module):
    def forward(self, x):
        size = list(x.size())
        return x.view(size[0], torch.IntTensor(size[1:]).prod())

def infer_shape(f, in_shape):
    return f(torch.zeros(*in_shape)).shape


def mlp(sizes, nl_hidden=nn.ReLU, nl_out=None):
    layers = [nn.Linear(sizes[0], sizes[1])]
    for i in range(1, len(sizes)-1):
        layers.append(nl_hidden())
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
    if nl_out:
        layers.append(n_out)
    return nn.Sequential(*layers)


def atari(in_channels=4, size=84):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        BatchFlatten()
    )
    n_conv_out = infer_shape(conv, [1, in_channels, size, size])[1]
    return nn.Sequential(
        conv,
        mlp([n_conv_out, 512])
    )

def lenet(in_channels, size=28):
    tanh = nn.Tanh()
    relu = nn.ReLU()
    pool = nn.MaxPool2d(2)
    conv = nn.Sequential(
        nn.Conv2d(in_channels, 6, 5), tanh, pool,
        nn.Conv2d(6, 16, 5), tanh, pool,
        nn.Conv2d(16, 120, 3), relu,
        BatchFlatten()
    )
    n_conv_out = infer_shape(conv, [1, in_channels, size, size])[1]
    return nn.Sequential(
        conv,
        mlp([n_conv_out, 84, 10])
    )

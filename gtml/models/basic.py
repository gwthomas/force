import torch.nn as nn


def mlp(sizes, activation=nn.ReLU):
    cg = []
    for i in range(len(sizes) - 1):
        cg.append(nn.Linear(sizes[i], sizes[i+1]))
        cg.append(activation())
    cg.pop(-1) # remove nonlinearity before final activation
    return nn.Sequential(*cg)
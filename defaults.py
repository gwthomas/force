import torch

BATCHSIZE = 64
EPSILON = 1e-8
DISCOUNT = 0.99
GAE_LAMBDA = 0.98

def OPTIMIZER(params):
    return torch.optim.Adam(params)

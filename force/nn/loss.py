import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


def supervised_loss(forward, criterion):
    return lambda x, y: criterion(forward(x), y)


class L2Loss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction='mean'):
        super(L2Loss, self).__init__(reduction=reduction)

    def forward(self, input, target):
        assert input.shape == target.shape, f'Shape mismatch: {input.shape} and {target.shape}'
        l2_dists = torch.dist(input, target)
        if self.reduction == 'mean':
            return l2_dists.mean()
        elif self.reduction == 'sum':
            return l2_dists.sum()
        elif self.reduction == 'none':
            return l2_dists
        else:
            raise NotImplementedError(f'Unknown reduction {self.reduction}')


NAMED_LOSS_FUNCTIONS = {
    'MSE': nn.MSELoss,
    'Huber': nn.HuberLoss,
    'L1': nn.L1Loss,
    'L2': L2Loss
}
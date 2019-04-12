import datetime
import os

import torch

from gtml.constants import DEFAULT_TIMESTAMP_FORMAT, DATASETS_DIR


def one_hot(labels):
    n = len(torch.unique(labels))
    return torch.eye(n)[labels]

def timestamp(format_string=DEFAULT_TIMESTAMP_FORMAT):
    now = datetime.datetime.now()
    return now.strftime(format_string)

def get_torch_dataset(which, train=True, test=True):
    import torchvision
    transform = torchvision.transforms.ToTensor()

    if which == 'mnist':
        data_class = torchvision.datasets.MNIST
    elif which == 'cifar10':
        data_class = torchvision.datasets.CIFAR10
    else:
        raise NotImplementedError('Dataset must be one of: mnist, cifar10')

    train_set, test_set = None, None
    if train:
        train_set = data_class(DATASETS_DIR, train=True, transform=transform,
                               download=True)
    if test:
        test_set = data_class(DATASETS_DIR, train=False, transform=transform,
                              download=True)
    return train_set, test_set

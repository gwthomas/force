from __future__ import print_function
import math
import os
import pdb

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Subset

from gtml.callbacks import *
import gtml.datasets as datasets
from gtml.config import Configuration, REQUIRED
from gtml.constants import DEVICE
from gtml.experiment import Experiment
import gtml.models.resnet as resnet
from gtml.train import EpochalMinimizer
from gtml.test import test
import gtml.util as util


cfg_template = Configuration([
    ('algorithm', ('sgd', 'adam'), 'adam'),
    ('init_lr', float, 0.1),
    ('weight_decay', float, 1e-4),
    ('momentum', float, 0.9),
    ('n_epochs', int, 200),
    ('batch_size', int, 128),
    ('load_exp', (str, None), None),
    ('load_index', (int, None), None),
    ('stage', ('1', '2', None), None),
    ('p', float, 0.5),
    ('linear', bool, False)
])


class LinearModel(nn.Module):
    def __init__(self, d_in, d_out):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = nn.functional.log_softmax(x)
        return x


def run(cfg, exp_name, train_set, test_set, model):
    parameters = model.parameters()
    criterion = torch.nn.CrossEntropyLoss()
    L = lambda x, y: criterion(model(x), y)

    if cfg.algorithm == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=cfg.init_lr,
                                    momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    elif cfg.algorithm == 'adam':
        optimizer = torch.optim.Adam(parameters, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError(cfg.algorithm)

    lr_scheduler = MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
    train = EpochalMinimizer(L, optimizer, train_set, cfg.batch_size)

    # Setup logging/data collection
    exp = Experiment(exp_name, serializables={
            'model': model,
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'train': train
    })

    load_exp = exp if cfg.load_exp is None else \
               Experiment(cfg.load_exp, serializables={
                    'model': model,
                    'train': train
               })

    if cfg.load_index is None:
        load_exp.load_latest()
    else:
        load_exp.load(index=cfg.load_index)

    if cfg.load_exp is not None:
        exp.data = load_exp.data

    def post_step_callback(steps_taken, loss):
        exp.log('Iteration {}: loss = {}', steps_taken, loss)
        exp.data.append('loss', loss)

    def post_epoch_callback(epochs_taken):
        exp.log('{} epochs completed. Evaluating...', epochs_taken)
        acc = test(model, test_set)
        exp.log('Accuracy: {}', acc)
        exp.data.append('test_accuracy', acc)
        exp.save(index=epochs_taken)

    train.add_callback('post_step', post_step_callback)
    train.add_callback('pre_epoch', lr_scheduler.step)
    train.add_callback('post_epoch', post_epoch_callback)

    train.run(cfg.n_epochs)


def main(cfg):
    if cfg.linear:
        train_set, test_set = datasets.load_mnist()
        model = LinearModel(28*28, 10)
    else:
        train_set, test_set = datasets.load_cifar10()
        model = resnet.PreActResNet18()

    if cfg.stage == None:
        exp_name = 'standard_{}'.format(cfg.algorithm)
    else:
        n = len(train_set)
        all_indices = torch.arange(n)
        n_partition1 = int(cfg.p * n)
        partition1 = all_indices[:n_partition1]
        partition2 = all_indices[n_partition1:]
        train_set = Subset(train_set, partition1 if cfg.stage == '1' else partition2)
        exp_name = 'stage{}_{}_{}'.format(cfg.stage, cfg.algorithm, cfg.p)

    if cfg.linear:
        exp_name = 'linear_' + exp_name

    run(cfg, exp_name, train_set, test_set, model)

if __name__ == '__main__':
    main(cfg_template.parse())

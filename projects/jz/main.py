from __future__ import print_function
import math
import os
import pdb

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Subset

from gtml.callbacks import *
import gtml.datasets as datasets
from gtml.config import Configuration, boolean, REQUIRED
from gtml.constants import DEVICE
from gtml.experiment import Experiment
import gtml.models.resnet as resnet
from gtml.optim import AdamW
from gtml.train import EpochalMinimizer
from gtml.test import test
import gtml.util as util


class PartitioningMinimizer(EpochalMinimizer):
    def __init__(self, compute_loss, optimizer, dataset, batch_size):
        EpochalMinimizer.__init__(self, compute_loss, optimizer)
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_index = -1
        self.partitions = [None, None]

    def _state_attrs(self):
        return EpochalMinimizer._state_attrs(self) + ['current_index']

    def current_partition(self):
        return self.partitions[self.current_index]

    def partition(self, p, randomize):
        n = len(self.dataset)
        all_indices = torch.randperm(n) if randomize else torch.arange(n)
        n_p1 = int(p * n)
        self.partitions[0] = Subset(self.dataset, all_indices[:n_p1])
        self.partitions[1] = Subset(self.dataset, all_indices[n_p1:])

    def use_partition(self, index):
        assert index in (0, 1)
        if index == self.current_index:
            return

        self.create_data_loader(self.partitions[index], batch_size=self.batch_size)
        self.current_index = index

    def switch_partition(self):
        self.use_partition(1 - self.current_index)


cfg_template = Configuration([
    ('exp_name', str, REQUIRED),
    ('dataset', ('cifar10', 'mnist', 'svhn'), REQUIRED),
    ('algorithm', ('sgd', 'adam', 'adamw'), REQUIRED),
    ('init_lr', (float, None), None),
    ('use_lr_schedule', boolean, True),
    ('weight_decay', float, 1e-4),
    ('momentum', float, 0.9),
    ('n_epochs', int, 200),
    ('batch_size', int, 128),
    ('load_exp', (str, None), None),
    ('load_index', (int, None), None),
    ('switch_period', (int, None), None),
    ('p', float, 0.5),
    ('randomize_partitions', boolean, False),
    ('seed', int, 0),
    ('debug_optimization_path_period', (int, None), None)
])


def main(cfg):
    util.set_random_seed(cfg.seed)

    if cfg.dataset == 'mnist':
        train_set, test_set = datasets.load_mnist()
    elif cfg.dataset == 'cifar10':
        train_set, test_set = datasets.load_cifar10()
    elif cfg.dataset == 'svhn':
        train_set, test_set = datasets.load_svhn()
    else:
        raise NotImplementedError

    model = resnet.PreActResNet18()
    if torch.cuda.is_available():
        print('CUDA is available :)')
        model.to(DEVICE)
    else:
        print('CUDA is not available :(')

    parameters = list(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    L = lambda x, y: criterion(model(x), y)

    if cfg.algorithm == 'sgd':
        init_lr = 0.1 if cfg.init_lr is None else float(cfg.init_lr)
        optimizer = SGD(parameters, lr=init_lr, momentum=cfg.momentum,
                        weight_decay=cfg.weight_decay)
    elif cfg.algorithm == 'adam':
        init_lr = 1e-3 if cfg.init_lr is None else float(cfg.init_lr)
        optimizer = Adam(parameters, lr=init_lr,
                         weight_decay=cfg.weight_decay)
    elif cfg.algorithm == 'adamw':
        init_lr = 1e-3 if cfg.init_lr is None else float(cfg.init_lr)
        optimizer = AdamW(parameters, lr=init_lr,
                          weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError(cfg.algorithm)

    train = PartitioningMinimizer(L, optimizer, train_set, cfg.batch_size)
    if cfg.switch_period is None:
        train.partition(1, False)
    else:
        train.partition(cfg.p, cfg.randomize_partitions)
    train.use_partition(0)

    serializables = {
        'model': model,
        'optimizer': optimizer,
        'train': train
    }

    if cfg.use_lr_schedule:
        lr_scheduler = MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        serializables['lr_scheduler'] = lr_scheduler
    else:
        lr_scheduler = None

    # Setup logging/data collection
    exp = Experiment(cfg.exp_name, serializables=serializables)

    load_exp = exp if cfg.load_exp is None else \
               Experiment(cfg.load_exp, serializables={
                    'model': model,
                    'train': train
               })

    if cfg.load_index is None:
        load_exp.load_latest()
    else:
        load_exp.load(index=cfg.load_index)

    if exp is not load_exp:
        exp.data = load_exp.data

    def evaluate(train_acc_key, test_acc_key):
        exp.log('Evaluating...')
        train_acc = test(model, train_set)
        test_acc = test(model, test_set)
        exp.log('Accuracy: train = {}, test = {}', train_acc, test_acc)
        exp.data.append(train_acc_key, train_acc)
        exp.data.append(test_acc_key, test_acc)

    if cfg.debug_optimization_path_period is not None:
        evaluate('debug_train_accuracy', 'debug_test_accuracy')

    def post_step_callback(steps_taken, loss):
        lr = optimizer.param_groups[0]['lr']
        exp.log('Iteration {}: lr = {}, loss = {}', steps_taken, lr, loss)
        exp.data.append('loss', loss)

        if cfg.debug_optimization_path_period is not None and steps_taken % cfg.debug_optimization_path_period == 0:
            evaluate('debug_train_accuracy', 'debug_test_accuracy')
            exp.save(index=steps_taken)

    def pre_epoch_callback(epochs_taken):
        if cfg.switch_period is not None and epochs_taken > 0 and \
                epochs_taken % cfg.switch_period == 0:
            train.switch_partition()
            exp.log('Switched to partition {}', train.current_index)

        if lr_scheduler is not None:
            lr_scheduler.step(epoch=epochs_taken)

    def post_epoch_callback(epochs_taken):
        exp.log('{} epochs completed.', epochs_taken)
        evaluate('train_accuracy', 'test_accuracy')
        exp.save(index=epochs_taken)

    train.add_callback('post_step', post_step_callback)
    train.add_callback('pre_epoch', pre_epoch_callback)
    train.add_callback('post_epoch', post_epoch_callback)
    train.run(cfg.n_epochs)


if __name__ == '__main__':
    main(cfg_template.parse())

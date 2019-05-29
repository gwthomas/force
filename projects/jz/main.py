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
from gtml.config import *
from gtml.constants import DEVICE
from gtml.experiment import Experiment
import gtml.models.resnet as resnet
from gtml.optim import AdamW
from gtml.train import EpochalMinimizer
from gtml.test import test
import gtml.util as util


class PartitioningMinimizer(EpochalMinimizer):
    def __init__(self, compute_loss, optimizer, dataset, p, randomize, batch_size):
        EpochalMinimizer.__init__(self, compute_loss, optimizer)
        self.dataset = dataset
        self.p = p
        self.randomize = randomize
        self.batch_size = batch_size
        self.current_index = -1
        self.partitions = [None, None]

    def current_partition(self):
        return self.partitions[self.current_index]

    def compute_partitions(self):
        n = len(self.dataset)
        all_indices = torch.randperm(n) if self.randomize else torch.arange(n)
        n_p1 = int(self.p * n)
        self.partitions[0] = Subset(self.dataset, all_indices[:n_p1])
        self.partitions[1] = Subset(self.dataset, all_indices[n_p1:])
        self.run_callbacks('computed_partitions')

    def use_partition(self, index):
        assert index in (0, 1)
        if index == self.current_index:
            return

        if self.randomize and index == 0:
            self.compute_partitions()
            
        self.create_data_loader(self.partitions[index], batch_size=self.batch_size)
        self.current_index = index
        self.run_callbacks('swapped_partition', self.current_index)


cfg_template = Config([
    RequiredItem('exp_name', str),
    RequiredItem('dataset', ('cifar10', 'mnist', 'svhn')),
    RequiredItem('algorithm', ('sgd', 'adam', 'adamw')),
    OptionalItem('init_lr', float),
    OptionalItem('drop_epoch', int),
    DefaultingItem('reset_optimizer', boolean, False),
    DefaultingItem('weight_decay', float, 1e-4),
    DefaultingItem('momentum', float, 0.9),
    DefaultingItem('n_epochs', int, 200),
    DefaultingItem('batch_size', int, 128),
    OptionalItem('load_exp', str),
    OptionalItem('load_index', int),
    DefaultingItem('eval_train', boolean, True),
    OptionalItem('p', float),
    OptionalItem('stage', ('1','2')),
    OptionalItem('swap_period', int),
    DefaultingItem('reset_on_swap', boolean, False),
    DefaultingItem('randomize_partitions', boolean, False),
    DefaultingItem('seed', int, 0),
    OptionalItem('debug_optimization_path_period', int)
])


def get_optimizer(cfg, parameters):
    if cfg.algorithm == 'sgd':
        init_lr = 0.1 if cfg.init_lr is None else cfg.init_lr
        return SGD(parameters, lr=init_lr, momentum=cfg.momentum,
                   weight_decay=cfg.weight_decay)
    elif cfg.algorithm == 'adam':
        init_lr = 1e-3 if cfg.init_lr is None else cfg.init_lr
        return Adam(parameters, lr=init_lr,
                    weight_decay=cfg.weight_decay)
    elif cfg.algorithm == 'adamw':
        init_lr = 1e-3 if cfg.init_lr is None else cfg.init_lr
        return AdamW(parameters, lr=init_lr,
                     weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError(cfg.algorithm)
        

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
    optimizer = get_optimizer(cfg, parameters)

    if cfg.p is None:
        train = EpochalMinimizer(L, optimizer)
        train.create_data_loader(train_set, batch_size=cfg.batch_size)
    else:
        p = cfg.p
        train = PartitioningMinimizer(L, optimizer, train_set, p, cfg.randomize_partitions, cfg.batch_size)
        train.compute_partitions()
        if cfg.stage is None:
            train.use_partition(0)
        else:
            train.use_partition(int(cfg.stage) - 1)
    
    serializables = {
        'model': model,
        'train': train
    }

    if cfg.drop_epoch is not None:
        lr_scheduler = MultiStepLR(optimizer, milestones=[cfg.drop_epoch], gamma=0.1)
        serializables['lr_scheduler'] = lr_scheduler
    else:
        lr_scheduler = None
        
    if not cfg.reset_optimizer:
        serializables['optimizer'] = optimizer

    # Setup logging/data collection
    exp = Experiment(cfg.exp_name, serializables=serializables)

    load_exp = exp if cfg.load_exp is None else \
               Experiment(cfg.load_exp, serializables=serializables)

    if cfg.load_index is None:
        load_exp.load_latest()
    else:
        load_exp.load(index=cfg.load_index)

    if exp is not load_exp:
        exp.data = load_exp.data

    def evaluate(train_acc_key, test_acc_key):
        exp.log('Evaluating...')
        test_acc = test(model, test_set)
        exp.data.append(test_acc_key, test_acc)
        if cfg.eval_train:
            train_acc = test(model, train_set)
            exp.data.append(train_acc_key, train_acc)
            exp.log('Accuracy: train = {}, test = {}', train_acc, test_acc)
        else:
            exp.log('Accuracy: test = {}', test_acc)
        

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
        if cfg.swap_period is not None:
            period = cfg.swap_period
            partition_index = (epochs_taken // period) % 2
            exp.log('Epochs taken = {}; partition index = {}', epochs_taken, partition_index)
            train.use_partition(partition_index)

        if lr_scheduler is not None:
            lr_scheduler.step(epoch=epochs_taken)

    def post_epoch_callback(epochs_taken):
        exp.log('{} epochs completed.', epochs_taken)
        evaluate('train_accuracy', 'test_accuracy')
        exp.save(index=epochs_taken)
        
    def on_swap(index):
        exp.log('Swapped to partition {}', index)
        if cfg.reset_on_swap:
            exp.log('Resetting optimizer')
            train.optimizer = get_optimizer(cfg, parameters)

    train.add_callback('post_step', post_step_callback)
    train.add_callback('pre_epoch', pre_epoch_callback)
    train.add_callback('post_epoch', post_epoch_callback)
    train.add_callback('computed_partitions', lambda: exp.log('Computed partitions'))
    train.add_callback('swapped_partition', on_swap)
    train.run(cfg.n_epochs)


if __name__ == '__main__':
    main(cfg_template.parse())

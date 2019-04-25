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


class Partitioner:
    def __init__(self, dataset):
        self.dataset = dataset
        
    


cfg_template = Configuration([
    ('exp_name', str, REQUIRED),
    ('dataset', ('cifar10', 'mnist', 'svhn'), REQUIRED),
    ('algorithm', ('sgd', 'adam', 'adamw'), REQUIRED),
    ('init_lr', (float, None), None),
    ('weight_decay', float, 1e-4),
    ('momentum', float, 0.9),
    ('n_epochs', int, 200),
    ('batch_size', int, 128),
    ('load_exp', (str, None), None),
    ('load_index', (int, None), None),
    ('stage', ('1', '2', None), None),
    ('p', float, 0.5),
    ('use_lr_schedule', boolean, True),
    ('seed', int, 0),
    ('debug_optimization_path_period', (int, None), None)
])


def run(cfg, train_set, test_set, model):
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
        
    train = EpochalMinimizer(L, optimizer, train_set, cfg.batch_size)
    
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
        
    def evaluate():
        exp.log('Evaluating...')
        acc = test(model, test_set)
        exp.log('Accuracy: {}', acc)
        return acc
    
    if cfg.debug_optimization_path_period is not None:
        acc = evaluate()
        exp.data.append('debug_test_accuracy', acc)

    def post_step_callback(steps_taken, loss):
        lr = optimizer.param_groups[0]['lr']
        exp.log('Iteration {}: lr = {}, loss = {}', steps_taken, lr, loss)
        exp.data.append('loss', loss)
        
        if cfg.debug_optimization_path_period is not None and steps_taken % int(cfg.debug_optimization_path_period) == 0:
            acc = evaluate()
            exp.data.append('debug_test_accuracy', acc)
            exp.save(index=steps_taken)
        
    def pre_epoch_callback(epochs_taken):
        if lr_scheduler is not None:
            lr_scheduler.step(epoch=epochs_taken)

    def post_epoch_callback(epochs_taken):
        exp.log('{} epochs completed.', epochs_taken)
        acc = evaluate()
        exp.data.append('test_accuracy', acc)
        exp.save(index=epochs_taken)

    train.add_callback('post_step', post_step_callback)
    train.add_callback('pre_epoch', pre_epoch_callback)
    train.add_callback('post_epoch', post_epoch_callback)
    
    train.run(cfg.n_epochs)


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

    if cfg.stage is not None:
        n = len(train_set)
        all_indices = torch.arange(n)
        n_partition1 = int(cfg.p * n)
        partition1 = all_indices[:n_partition1]
        partition2 = all_indices[n_partition1:]
        train_set = Subset(train_set, partition1 if cfg.stage == '1' else partition2)

    run(cfg, train_set, test_set, model)

if __name__ == '__main__':
    main(cfg_template.parse())

from __future__ import print_function
import math
import os
import pdb

import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import torchvision.models as models

from gtml.callbacks import *
from gtml.config import Config, Require
from gtml.core import Enumeration
from gtml.experiment import Experiment
from gtml.train import EpochalMinimizer
from gtml.test import test
import gtml.util as util


DATASET_OPTIONS = Enumeration(['mnist', 'cifar10'])
LR_SCHEDULE_OPTIONS = Enumeration(['multi_step', 'cosine_annealing'])

cfg = Config({
    'dataset': Require(DATASET_OPTIONS),
    'lr_schedule': Require(LR_SCHEDULE_OPTIONS),
    'n_epochs': 200,
    'batch_size': 128
})


def main(load):
    # Load data
    train_set, test_set = util.get_torch_dataset(cfg['dataset'])
    n = len(train_set)
    steps_per_epoch = int(math.ceil(n / cfg['batch_size']))

    # Instantiate model and training procedure
    model = models.resnet18(num_classes=10)
    criterion = torch.nn.CrossEntropyLoss()
    L = lambda x, y: criterion(model(x), y)
    optimizer = torch.optim.Adam(model.parameters())

    if cfg['lr_schedule'] == 'multi_step':
        lr_scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.1)
    elif cfg['lr_schedule'] == 'cosine_annealing':
        T_max = steps_per_epoch * cfg['n_epochs']
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-3)

    train = EpochalMinimizer(L, optimizer, train_set, cfg['batch_size'])

    # Setup logging/data collection
    exp = Experiment('reproduce', serializables={
        'model': model,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'train': train
    })

    def post_step_callback(steps_taken, loss):
        exp.log('Iteration {}: loss = {}', steps_taken, loss)
        exp.data.append('loss', loss)

    def post_epoch_callback(epochs_taken):
        exp.log('{} epochs completed. Evaluating...', epochs_taken)
        acc = test(model, test_set)
        exp.log('Accuracy: {}', acc)
        exp.data['test_accuracy'].append(acc)
        exp.save(index=epochs_taken)

    train.add_callback('post_step', post_step_callback)
    train.add_callback('pre_epoch', lr_scheduler.step)
    train.add_callback('post_epoch', post_epoch_callback)

    if load is not None:
        exp.load(index=load)

    # Train
    train.run(cfg['n_epochs'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--lr_schedule', type=str, default='cosine_annealing')
    parser.add_argument('--load', default=None)
    args = parser.parse_args()
    cfg.update(vars(args))
    main(args.load)

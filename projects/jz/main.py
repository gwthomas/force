from __future__ import print_function
import math
import os
import pdb

import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import torchvision.models as models

from gtml.callbacks import *
from gtml.config import Configuration
from gtml.experiment import Experiment
from gtml.train import EpochalMinimizer
from gtml.test import test
import gtml.util as util


cfg_template = Configuration([
    ('dataset', ['mnist', 'cifar10'], None),
    ('algorithm', ['sgd', 'adam'], 'adam'),
    ('lr_schedule', ['multi_step', 'cosine_annealing'], 'multi_step'),
    ('weight_decay', float, 1e-4),
    ('momentum', float, 0.9),
    ('n_epochs', int, 200),
    ('batch_size', int, 128),
    ('load', int, -1)
])

def main(cfg):
    # Load data
    train_set, test_set = util.get_torch_dataset(cfg.dataset)
    n = len(train_set)
    steps_per_epoch = int(math.ceil(n / cfg.batch_size))

    # Instantiate model and training procedure
    model = models.resnet18(num_classes=10)

    parameters = model.parameters()
    criterion = torch.nn.CrossEntropyLoss()
    L = lambda x, y: criterion(model(x), y)

    if cfg.algorithm == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=0.1, momentum=args.momentum,
                                    weight_decay=cfg.weight_decay)
    elif cfg.algorithm == 'adam':
        optimizer = torch.optim.Adam(parameters, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError(cfg.algorithm)

    if cfg.lr_schedule == 'multi_step':
        lr_scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.1)
    elif cfg.lr_schedule == 'cosine_annealing':
        T_max = steps_per_epoch * cfg.n_epochs
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-3)
    else:
        raise NotImplementedError(cfg.algorithm)

    train = EpochalMinimizer(L, optimizer, train_set, cfg.batch_size)

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
        exp.data.append('test_accuracy', acc)
        exp.save(index=epochs_taken)

    train.add_callback('post_step', post_step_callback)
    train.add_callback('pre_epoch', lr_scheduler.step)
    train.add_callback('post_epoch', post_epoch_callback)

    if cfg.load == -1:
        exp.load_latest()
    else:
        exp.load(index=cfg.load)

    if torch.cuda.is_available():
        exp.log('CUDA is available')
        model = model.to('cuda:0')
    else:
        exp.log('CUDA is not available :(')

    train.run(cfg.n_epochs)


if __name__ == '__main__':
    main(cfg_template.parse())

import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from gtml.callbacks import *
from gtml.config import *
from gtml.core import Serializable
from gtml.constants import DEVICE
import gtml.datasets as datasets
from gtml.experiment import Experiment
from gtml.models.basic import mlp
from gtml.optim import AdamW
from gtml.train import Minimizer
from gtml.test import test
import gtml.util as util

import pdb


cfg_template = Config([
    RequiredItem('exp_name', str),
    RequiredItem('algorithm', ('sgd', 'adam', 'adamw')),
    DefaultingItem('chunk_size', int, 50000),
    OptionalItem('init_lr', float),
    DefaultingItem('weight_decay', float, 1e-4),
    DefaultingItem('momentum', float, 0.9),
    DefaultingItem('batch_size', int, 1),
    DefaultingItem('seed', int, 0),
])


def get_optimizer(cfg, parameters):
    if cfg.algorithm == 'sgd':
        init_lr = 0.01 if cfg.init_lr is None else cfg.init_lr
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

    train_set, test_set = datasets.load_emnist('bymerge',
            other_transforms=[transforms.Lambda(lambda x: x.view(-1))])
    n_examples = len(train_set)
    order = torch.randperm(n_examples)
    train_set = Subset(train_set, order)

    model = mlp([28*28, 500, 500, 47])
    if torch.cuda.is_available():
        print('CUDA is available :)')
        model.to(DEVICE)
    else:
        print('CUDA is not available :(')

    parameters = list(model.parameters())
    optimizer = get_optimizer(cfg, parameters)
    criterion = torch.nn.CrossEntropyLoss()
    L = lambda x, y: criterion(model(x), y)

    serializables = {'model': model}
    exp = Experiment(cfg.exp_name, serializables=serializables)

    def evaluate(steps_taken, loss):
        chunks_completed = steps_taken // cfg.chunk_size
        exp.log('Processed {} chunks ({} examples). Evaluating...', chunks_completed, steps_taken)
        test_acc = test(model, test_set)
        exp.log('Test accuracy: {}', test_acc)
        chunk_accs = []
        for chunk_index in range(chunks_completed):
            indices = torch.arange(cfg.chunk_size * chunk_index,
                                   cfg.chunk_size * (chunk_index+1))
            chunk = Subset(train_set, indices)
            chunk_acc = test(model, chunk)
            exp.log('Chunk {} accuracy: {}', chunk_index, chunk_acc)
            chunk_accs.append(chunk_acc)
        exp.data.append('test_acc', test_acc)
        exp.data.append('chunk_accs', chunk_accs)
        exp.log('Saving')
        exp.save(chunks_completed)

    train = Minimizer(L, optimizer)
    train.add_callback('post_step', Periodic(cfg.chunk_size, evaluate))

    data_loader = DataLoader(train_set, batch_size=cfg.batch_size,
                             shuffle=True, pin_memory=True,
                             num_workers=2)
    for x, y in data_loader:
        train.step(x, y)


if __name__ == '__main__':
    main(cfg_template.parse())

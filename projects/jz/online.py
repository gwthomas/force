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
from gtml.models.basic import mlp, convnet
from gtml.optim import AdamW
from gtml.train import Minimizer
from gtml.test import test
import gtml.util as util

import pdb


cfg_template = Config([
    RequiredItem('exp_name', str),
    RequiredItem('dataset', ('mnist', 'fashion-mnist', 'emnist-byclass', 'emnist-bymerge')),
    RequiredItem('algorithm', ('sgd', 'adam', 'adamw')),
    DefaultingItem('n_chunks', int, 10),
    OptionalItem('init_lr', float),
    DefaultingItem('weight_decay', float, 1e-4),
    DefaultingItem('momentum', float, 0.9),
    DefaultingItem('steps_per_example', int, 1),
    DefaultingItem('seed', int, 0),
])


def get_optimizer(cfg, parameters):
    if cfg.algorithm == 'sgd':
        init_lr = 1e-3 if cfg.init_lr is None else cfg.init_lr
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

            #other_transforms=[transforms.Lambda(lambda x: x.view(-1))])
    if cfg.dataset == 'mnist':
        train_set, test_set = datasets.load_mnist()
        n_classes = 10
    elif cfg.dataset == 'fashion-mnist':
        train_set, test_set = datasets.load_fashion_mnist()
        n_classes = 10
    elif cfg.dataset == 'emnist-byclass':
        train_set, test_set = datasets.load_emnist('byclass')
        n_classes = 62
    elif cfg.dataset == 'emnist-bymerge':
        train_set, test_set = datasets.load_emnist('bymerge')
        n_classes = 47
        
    n_examples = len(train_set)    
    order = torch.randperm(n_examples)
    train_set = Subset(train_set, order)
    data_loader = DataLoader(train_set, batch_size=1,
                             shuffle=False, pin_memory=True,
                             num_workers=2)

    #model = mlp([28*28, 500, 500, n_classes])
    model = convnet((1,28,28), [(6,5), (16,5)], [500, n_classes])
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
    
    chunk_size = n_examples // cfg.n_chunks
    exp.log('{} chunks, each of size {}', cfg.n_chunks, chunk_size)

    def post_step(steps_taken, loss):
        if steps_taken % cfg.steps_per_example != 0:
            return
        examples_completed = steps_taken // cfg.steps_per_example
        if examples_completed % chunk_size != 0:
            return
        chunks_completed = examples_completed // chunk_size
        
        exp.log('Processed {} chunks ({} examples). Evaluating...', chunks_completed, examples_completed)
        test_acc = test(model, test_set)
        exp.log('Test accuracy: {}', test_acc)
        chunk_accs = []
        for chunk_index in range(chunks_completed):
            end_index = min(chunk_size * (chunk_index+1), n_examples)
            indices = torch.arange(chunk_size * chunk_index, end_index)
            chunk = Subset(train_set, indices)
            chunk_acc = test(model, chunk)
            exp.log('Chunk {} accuracy: {}', chunk_index, chunk_acc)
            chunk_accs.append(chunk_acc)
        exp.data.append('test_acc', test_acc)
        exp.data.append('chunk_accs', chunk_accs)
        exp.log('Saving')
        exp.save(chunks_completed)

    train = Minimizer(L, optimizer)
    train.add_callback('post_step', post_step)
    
    for x, y in data_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        for _ in range(cfg.steps_per_example):
            train.step(x, y)


if __name__ == '__main__':
    main(cfg_template.parse())

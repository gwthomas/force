from __future__ import print_function
import math
import pdb

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import Subset
import torchvision.transforms as transforms

from gtml.callbacks import *
from gtml.config import *
from gtml.core import Serializable
from gtml.constants import DEVICE
import gtml.datasets as datasets
from gtml.experiment import Experiment
from gtml.models.basic import mlp, convnet
from gtml.optim import AdamW
from gtml.train import LossFunction, EpochalMinimizer
from gtml.test import test
import gtml.util as util


class EWC(Serializable):
    def __init__(self, model, lamb, use_empirical_fisher):
        self.model = model
        self.params = {name: param for name, param in model.named_parameters() if param.requires_grad}
        self.lamb = lamb
        self.use_empirical_fisher = use_empirical_fisher
        self.completed = []

    def _state_attrs(self):
        return ['completed']

    def task_complete(self, train_set, n_classes=10):
        param_vals, fish_diags = {}, {}
        for name, param in self.params.items():
            param_vals[name] = param.data.clone()
            fish_diags[name] = torch.zeros_like(param)

        n = len(train_set)
        
        for x, y in train_set:
            x = x.unsqueeze(0).to(DEVICE)
            log_probs = self.model(x).log_softmax(1)[0]
            
            if self.use_empirical_fisher:
                # Use true label y instead of expectation over p(y|x)
                self.model.zero_grad()
                log_probs[y].backward()
                for name, param in self.params.items():
                    fish_diags[name] += 1./n * param.grad**2
            else:
                probs = log_probs.exp()
                for y in range(n_classes):
                    self.model.zero_grad()
                    retain = (y != n_classes-1)
                    log_probs[y].backward(retain_graph=retain)
                    p_y = probs[y].item()
                    for name, param in self.params.items():
                        fish_diags[name] += 1./n * p_y * param.grad**2    

        self.completed.append((param_vals, fish_diags))

    def compute_penalty(self):
        l = 0
        for name, param in self.params.items():
            for param_vals, fish_diags in self.completed:
                l = l + self.lamb / 2 * (fish_diags[name] * (param - param_vals[name])**2).sum()
        return l


cfg_template = Config([
    RequiredItem('exp_name', str),
    RequiredItem('task', ('permuted-mnist', 'fashion-mnist-split')),
    RequiredItem('n_tasks', int),
    RequiredItem('algorithm', ('sgd', 'adam', 'adamw')),
    OptionalItem('init_lr', float),
    OptionalItem('load_exp', str),
    OptionalItem('load_index', int),
    DefaultingItem('reset_optimizer', boolean, True),
    DefaultingItem('weight_decay', float, 1e-4),
    DefaultingItem('momentum', float, 0.9),
    DefaultingItem('n_epochs', int, 100), # epochs per task
    DefaultingItem('batch_size', int, 128),
    DefaultingItem('seed', int, 0),
    OptionalItem('ewc_lambda', float),
    DefaultingItem('use_empirical_fisher', boolean, False),
    DefaultingItem('skip_tasks', int, 0)
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


def load_permuted_mnist(permutation):
    #permutation = torch.randperm(28*28)
    permute = transforms.Lambda(lambda x: x.view(-1)[permutation])
    return datasets.load_mnist(other_transforms=[permute])

def create_permuted_mnist_tasks(n_tasks):
    orig_train, orig_test = datasets.load_mnist(other_transforms=[
            transforms.Lambda(lambda x: x.view(-1))
    ])
    tasks = [(orig_train, orig_test)]

    permutations = torch.load('mnist_permutations.pt')
    for i in range(n_tasks-1):
        train_set, test_set = load_permuted_mnist(permutations[i])
        tasks.append((train_set, test_set))
    return tasks


def split_by_classes(dataset, class_splits):
    splits = [[] for _ in class_splits]
    for data_index, (x, y) in enumerate(dataset):
        for split_index, split_classes in enumerate(class_splits):
            if y in split_classes:
                splits[split_index].append(data_index)
                break
    return [Subset(dataset, split) for split in splits]

def create_fashion_mnist_split_tasks():
    orig_train, orig_test = datasets.load_fashion_mnist()
    class_splits = [[0,1,2,3,4], [5,6,7,8,9]]
    task1_train, task2_train = split_by_classes(orig_train, class_splits)
    task1_test, task2_test = split_by_classes(orig_test, class_splits)
    return [(task1_train, task1_test), (task2_train, task2_test)]
    

#def create_mnist_split_tasks(n_tasks):
#    orig_train, orig_test = datasets.load_mnist(other_transforms=[
#            transforms.Lambda(lambda x: x.view(-1))
#    ])
#    n = len(orig_train)
#    all_indices = torch.randperm(n)
#    all_task_indices = torch.chunk(all_indices, n_tasks)
#    tasks = []
#    for task_indices in all_task_indices:
#        train_set = Subset(orig_train, task_indices)
#        tasks.append((train_set, train_set))
#    return tasks


def main(cfg):
    util.set_random_seed(cfg.seed)
    
    if cfg.task == 'permuted-mnist':
        tasks = create_permuted_mnist_tasks(cfg.n_tasks)
        model = mlp([28*28, 100, 100, 10])
    elif cfg.task == 'fashion-mnist-split':
        assert cfg.n_tasks == 2
        tasks = create_fashion_mnist_split_tasks()
        model = convnet((1,28,28), [(6,5), (6,5)], [100, 10])
    
    print('Model has', util.count_parameters(model), 'parameters')
        
    if torch.cuda.is_available():
        print('CUDA is available :)')
        model.to(DEVICE)
    else:
        print('CUDA is not available :(')

    parameters = list(model.parameters())
    optimizer = get_optimizer(cfg, parameters)
    criterion = torch.nn.CrossEntropyLoss()

    serializables = {
        'model': model
    }

    if cfg.ewc_lambda is None:
        ewc = None
    else:
        ewc = EWC(model, cfg.ewc_lambda, cfg.use_empirical_fisher)
        serializables['ewc'] = ewc

    exp = Experiment(cfg.exp_name, serializables=serializables)
    if cfg.load_exp is not None:
        exp.load_other(cfg.load_exp, index=cfg.load_index)
        
    def post_step_callback(steps_taken, loss, terms):
        for name, value in terms.items():
            exp.data.append(name, value)

    def post_epoch_callback(epochs_taken):
        exp.log('Completed epoch {} of task {}. Evaluating...', epochs_taken, tasks_completed)
        results = []
        for task_index in range(tasks_completed + 1):
            result = test(model, tasks[task_index][1])
            results.append(result)
        exp.log('Results: {}', results)
        exp.data.append('results', results)
        global_epochs = tasks_completed * cfg.n_epochs + epochs_taken
        exp.save(global_epochs)
    
    L = LossFunction()
    L.add_supervised_term('supervised', model, criterion)
    if ewc is not None:
        L.add_term('ewc_penalty', lambda _: ewc.compute_penalty())

    tasks_completed = cfg.skip_tasks
    while tasks_completed < cfg.n_tasks:
        train_set, test_set = tasks[tasks_completed]
        train = EpochalMinimizer(L, optimizer)
        train.create_data_loader(train_set, batch_size=cfg.batch_size)
        train.add_callback('post_step', post_step_callback)
        train.add_callback('post_epoch', post_epoch_callback)
        train.run(cfg.n_epochs)
        tasks_completed += 1
        if ewc is not None and tasks_completed < cfg.n_tasks:
            exp.log('Computing information for EWC...')
            ewc.task_complete(train_set)

        # Save again for the EWC parameters
        exp.save(tasks_completed * cfg.n_epochs)

        if cfg.reset_optimizer:
            optimizer = get_optimizer(cfg, parameters)

if __name__ == '__main__':
    main(cfg_template.parse())

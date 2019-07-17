import torch
from torch.optim.lr_scheduler import MultiStepLR

import gtml.datasets as datasets
from gtml.constants import DEVICE
import gtml.models.resnet as resnet
from gtml.train import LossFunction, EpochalMinimizer
from gtml.test import test
import gtml.util as util
from gtml.workflow.config import *


config_info = Config([
    ConfigItem('dataset', ('cifar10', 'mnist', 'svhn'), REQUIRED),
    ConfigItem('init_lr', float, 0.1),
    ConfigItem('batch_size', int, 128),
    ConfigItem('weight_decay', float, 1e-4),
    ConfigItem('momentum', float, 0.9),
    ConfigItem('n_epochs', int, 200),
    ConfigItem('drop_epoch', int, OPTIONAL),
    ConfigItem('eval_train', bool, True)
])


def main(exp, cfg):
    if cfg['dataset'] == 'mnist':
        train_set, test_set = datasets.load_mnist()
    elif cfg['dataset'] == 'cifar10':
        train_set, test_set = datasets.load_cifar10()
    elif cfg['dataset'] == 'svhn':
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
    L = LossFunction()
    L.add_supervised_term(model, criterion)
    optimizer = torch.optim.SGD(parameters, cfg['init_lr'],
                                momentum=cfg['momentum'],
                                weight_decay=cfg['weight_decay'])

    train = EpochalMinimizer(L, optimizer)
    train.create_data_loader(train_set, batch_size=cfg['batch_size'])

    if cfg['drop_epoch'] is not None:
        lr_scheduler = MultiStepLR(optimizer, milestones=[cfg['drop_epoch']], gamma=0.1)
    else:
        lr_scheduler = None

    exp.register_serializables({
        'model': model,
        'train': train,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler
    })

    def evaluate():
        exp.log('Evaluating...')
        test_acc = test(model, test_set)
        exp.data.append('test_acc', test_acc)
        if cfg['eval_train']:
            train_acc = test(model, train_set)
            exp.data.append('train_acc', train_acc)
            exp.log('Accuracy: train = {}, test = {}', train_acc, test_acc)
        else:
            exp.log('Accuracy: test = {}', test_acc)

    def post_step_callback(steps_taken, loss, terms):
        lr = optimizer.param_groups[0]['lr']
        exp.log('Iteration {}: lr = {}, loss = {}', steps_taken, lr, loss)
        exp.data['loss'].append(loss)

    def pre_epoch_callback(epochs_taken):
        if lr_scheduler is not None:
            lr_scheduler.step(epoch=epochs_taken)

    def post_epoch_callback(epochs_taken):
        exp.log('{} epochs completed.', epochs_taken)
        evaluate()
        exp.save(index=epochs_taken)

    train.add_callback('post_step', post_step_callback)
    train.add_callback('pre_epoch', pre_epoch_callback)
    train.add_callback('post_epoch', post_epoch_callback)
    train.run(cfg['n_epochs'])

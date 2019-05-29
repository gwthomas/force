import torch

from gtml.config import Configuration, boolean, REQUIRED
from gtml.constants import DEVICE
import gtml.datasets as datasets
from gtml.experiment import Experiment
import gtml.models.resnet as resnet
from gtml.test import test


TRAIN_KEY = 'train_accuracy'
TEST_KEY = 'test_accuracy'


cfg_template = Configuration([
    ('exp_name', str, REQUIRED),
    ('dataset', ('cifar10', 'svhn'), REQUIRED),
    ('train', boolean, False),
    ('test', boolean, False)
])


def main(cfg):
    if not cfg.train and not cfg.test:
        print('Neither train nor test was passed. Exiting')
        exit(1)
    
    if cfg.dataset == 'cifar10':
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
        
    train_accs = []
    test_accs = []

    serializables = {'model': model}
    exp = Experiment(cfg.exp_name, serializables=serializables)
    for ckpt_index in exp.list_checkpoints():
        exp.load(ckpt_index)
        if cfg.train:
            train_acc = test(model, train_set)
            exp.log('Checkpoint {}: train accuracy = {}', ckpt_index, train_acc)
            train_accs.append(train_acc)
            exp.data[TRAIN_KEY] = list(train_accs)
        if cfg.test:
            test_acc = test(model, test_set)
            exp.log('Checkpoint {}: test accuracy = {}', ckpt_index, test_acc)
            test_accs.append(test_acc)
            exp.data[TEST_KEY] = list(test_accs)
        exp.save(ckpt_index)

if __name__ == '__main__':
    main(cfg_template.parse())
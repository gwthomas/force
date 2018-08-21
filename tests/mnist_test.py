from __future__ import print_function

import matplotlib.pyplot as plt
import torch; nn = torch.nn; F = nn.functional; optim = torch.optim

from gtml.common.datasets import load_mnist, load_cifar
from gtml.defaults import *
from gtml.nn.proto import *
from gtml.nn.opt import SupervisedLearning

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar', action='store_true')
    args = parser.parse_args()
    cifar = args.cifar
    print('Training on', 'CIFAR' if cifar else 'MNIST')

    Xtrain, Ytrain, Xtest, Ytest = load_cifar('cifar10') if cifar else load_mnist('mnist')

    net = LeNet(3 if cifar else 1)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(net.parameters())
    train = SupervisedLearning(net, criterion)

    def print_loss(train, **kwargs):
        if train.steps_taken % 100 == 0:
            print('Iteration:', train.steps_taken, 'Loss:', train.losses[-1])

    def print_accuracy(train, **kwargs):
        output = net(Xtest)
        values, indices = torch.max(output, 1)
        accuracy = (indices == Ytest).float().mean().data[0]
        print('Epoch:', train.epochs_taken, 'Accuracy:', accuracy)

    train.add_callback('post-step', print_loss)
    train.add_callback('post-epoch', print_accuracy)
    train.run(Xtrain, Ytrain, 10)
    plt.plot(train.losses)
    plt.show()

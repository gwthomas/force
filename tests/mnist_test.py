from __future__ import print_function

import matplotlib.pyplot as plt
import torch; nn = torch.nn; F = nn.functional; optim = torch.optim

from gtml.datasets import load_mnist, load_cifar
from gtml.defaults import *
import gtml.nn.proto as proto
from gtml.nn.opt import SupervisedLearning

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar', action='store_true')
    args = parser.parse_args()
    if args.cifar:
        print('CIFAR')
        Xtrain, Ytrain, Xtest, Ytest = load_cifar('cifar10')
        channels = 3
        size = 32
    else:
        print('MNIST')
        Xtrain, Ytrain, Xtest, Ytest =  load_mnist('mnist')
        channels = 1
        size = 28

    net = proto.lenet(channels, size)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
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

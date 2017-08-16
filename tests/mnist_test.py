from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from gtml.common.datasets import load_mnist, load_cifar
from gtml.common.tf import init_sess
from gtml.defaults import *
import gtml.nn.layers as layers
import gtml.nn.proto as proto
from gtml.train.loss import *
from gtml.train.supervised import SupervisedLearning


def build(in_shape):
    x_in = tf.placeholder(FLOAT_T, in_shape)
    y_in = tf.placeholder(INT_T, [None])
    y_oh = tf.one_hot(y_in, 10)
    with tf.variable_scope('conv'):
        conv_out = proto.convnet(x_in,
                filters=[(64, 3), (32, 3)],
                strides=[2, 2])
    mlp_in = layers.flatten(conv_out)
    with tf.variable_scope('mlp'):
        logits = proto.mlp(mlp_in, [200, 100, 10])
    loss = softmax_cross_entropy(y_oh, logits)
    predictions = tf.argmax(logits, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_in), FLOAT_T), name='accuracy')
    return x_in, y_in, loss


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar', action='store_true')
    args = parser.parse_args()
    cifar = args.cifar
    print('Training on', 'CIFAR' if cifar else 'MNIST')

    in_shape = [None, 32, 32, 3] if cifar else [None, 28, 28, 1]
    x_in, y_in, loss = build(in_shape)

    def print_loss(train, **kwargs):
        if train.steps_taken % 100 == 0:
            print('Iteration:', train.steps_taken, 'Loss:', train.losses[-1])

    def print_accuracy(train, **kwargs):
        feed_dict = {x_in: Xtest, y_in: Ytest}
        print('Epoch:', train.epochs_taken, 'Accuracy:', sess.run('accuracy:0', feed_dict=feed_dict))

    train = SupervisedLearning(loss, x_in, y_in)
    train.add_callback('post-step', print_loss)
    train.add_callback('post-epoch', print_accuracy)

    Xtrain, Ytrain, Xtest, Ytest = load_cifar('cifar10') if cifar else load_mnist('mnist')

    with tf.Session() as sess:
        init_sess(sess)
        train.run(Xtrain, Ytrain, 10)

    plt.plot(train.losses)
    plt.show()

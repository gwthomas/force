from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf

from gtml.common.datasets import load_mnist, load_cifar
from gtml.common.tf import init_sess
from gtml.defaults import *
import gtml.nn.layers as layers
import gtml.nn.proto as proto
from gtml.train.loss import *

from gtml.experiments.noisy.noisy_training import NoisySupervisedLearning


def build(in_shape):
    x_in = tf.placeholder(FLOAT_T, in_shape)
    y_in = tf.placeholder(INT_T, [None])
    y_oh = tf.one_hot(y_in, 10)
    with tf.variable_scope('conv'):
        conv_out, conv_vars = proto.convnet(x_in,
                filters=[(64, 3), (32, 3)],
                strides=[2, 2])
    mlp_in, _ = layers.flatten(conv_out)
    with tf.variable_scope('mlp'):
        logits, mlp_vars = proto.mlp(mlp_in, [200, 100, 10])
    loss = softmax_cross_entropy(y_oh, logits)
    predictions = tf.argmax(logits, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_in), FLOAT_T), name='accuracy')
    return x_in, y_in, loss, conv_vars + mlp_vars


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scale', metavar='SCALE', type=float, default=1e-8)
    parser.add_argument('--cifar', action='store_true')
    args = parser.parse_args()
    scale = args.scale
    cifar = args.cifar
    print('Scale', scale)
    print('Training on', 'CIFAR' if cifar else 'MNIST')

    in_shape = [None, 32, 32, 3] if cifar else [None, 28, 28, 1]
    x_in, y_in, loss, variables = build(in_shape)

    def print_loss(_, itr, loss, **kwargs):
        if itr % 100 == 0:
            print('Itr:', itr, 'Loss:', loss)

    def print_accuracy(train, epoch, sess, **kwargs):
        feed_dict = {x_in: Xtest, y_in: Ytest}
        print('Epoch:', epoch, 'Accuracy:', sess.run('accuracy:0', feed_dict=feed_dict))

    opt = tf.train.AdamOptimizer()
    # opt = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9)
    train = NoisySupervisedLearning(opt, loss, x_in, y_in, scale)
    train.add_callback('post-iteration', print_loss)
    train.add_callback('post-epoch', print_accuracy)

    Xtrain, Ytrain, Xtest, Ytest = load_cifar('cifar10') if cifar else load_mnist('mnist')

    with tf.Session() as sess:
        init_sess(sess)
        train.run(Xtrain, Ytrain, 10)
    plt.plot(train.losses)
    plt.show()

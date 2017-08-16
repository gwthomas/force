import numpy as np
import tensorflow as tf

from gtml.nn.variable import get_default_variable_manager


def dense(input, size, nonlinearity, name=None, variable_manager=get_default_variable_manager()):
    assert len(input.shape) == 2
    prev_size = input.shape[1]
    W_name, b_name = ('W', 'b') if name is None else ('W_{}'.format(name), 'b_{}'.format(name))
    W = variable_manager.get_variable(W_name, [prev_size, size], kind='dense')
    b = variable_manager.get_variable(b_name, [size], kind='bias')
    pre = tf.nn.xw_plus_b(input, W, b)
    return pre if nonlinearity is None else nonlinearity(pre)

def conv2d(input, n_filters, filter_size, stride, nonlinearity, padding='SAME',
        name=None, variable_manager=get_default_variable_manager()):
    assert len(input.shape) == 4
    n_channels = input.shape[-1]
    W_name, b_name = ('W', 'b') if name is None else ('W_{}'.format(name), 'b_{}'.format(name))
    W = variable_manager.get_variable(W_name, [filter_size, filter_size, n_channels, n_filters], kind='conv')
    b = variable_manager.get_variable(b_name, [n_filters], kind='bias')
    strides = [1, stride, stride, 1]
    pre = tf.nn.conv2d(input, W, strides, padding) + b
    return pre if nonlinearity is None else nonlinearity(pre)

def flatten(input):
    return tf.reshape(input, [-1, int(np.prod(input.shape[1:]))])

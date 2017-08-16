import gym
import itertools as it
import tensorflow as tf

import gtml.nn.layers as layers
from gtml.nn.variable import get_default_variable_manager


def mlp(input, sizes, nl=tf.nn.relu, output_nl=None, variable_manager=get_default_variable_manager()):
    sofar = input
    for l, size in enumerate(sizes):
        nonlinearity = nl if l + 1 < len(sizes) else output_nl
        name = 'mlp_layer_{}'.format(l)
        sofar = layers.dense(sofar, size, nonlinearity, name=name, variable_manager=variable_manager)
    return sofar

def convnet(input, filters, strides, poolings=None, nl=tf.nn.relu, variable_manager=get_default_variable_manager()):
    poolings = [None]*len(filters) if poolings is None else poolings
    sofar = input
    for l, filter, stride, pooling in zip(it.count(), filters, strides, poolings):
        n_filters, filter_size = filter
        name = 'conv_layer_{}'.format(l)
        sofar = layers.conv2d(sofar, n_filters, filter_size, stride=stride,
                nonlinearity=nl, name=name, variable_manager=variable_manager)
        # if pooling is not None:
        #     sofar = layers.max_pool(???)
    return sofar

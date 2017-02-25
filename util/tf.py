import numpy as np
import tensorflow as tf


def init_sess(sess=None):
    if sess is None:
        sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())

def flatten(tensor, num_leading_axes=1):
    shape = tensor.get_shape().as_list()
    size = np.prod(shape[num_leading_axes:])
    return tf.reshape(tensor, [-1]*num_leading_axes+[size])

def squared_error(x, y):
    return tf.reduce_sum((x - y)**2)

def mean_squared_error(x, y):
    return tf.reduce_mean((x - y)**2)

# Adapted from http://stackoverflow.com/questions/37026425/elegant-way-to-select-one-element-per-row-in-tensorflow
def selection_slice(matrix, idx, n):
    # return matrix[tf.range(n),idx]
    return tf.gather_nd(matrix, tf.transpose(tf.stack([tf.range(n), idx])))

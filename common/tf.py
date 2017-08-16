import numpy as np
import os
import tensorflow as tf


def get_sess(sess=None):
    return tf.get_default_session() if sess is None else sess

def init_sess(sess=None):
    return get_sess(sess).run(tf.global_variables_initializer())

def flatten(tensor, num_leading_axes=0):
    shape = tensor.get_shape().as_list()
    collapsed_dims = shape[num_leading_axes:]
    size = -1 if None in collapsed_dims else np.prod(collapsed_dims)
    shape_tf = tf.shape(tensor)
    new_shape = tf.concat([shape_tf[:num_leading_axes], [size]], 0)
    retval = tf.reshape(tensor, shape=new_shape)
    return retval

def duplicate_placeholder(placeholder):
    return tf.placeholder(placeholder.dtype, shape=placeholder.shape)

# Adapted from http://stackoverflow.com/questions/37026425/elegant-way-to-select-one-element-per-row-in-tensorflow
# def selection_slice(matrix, idx, n):
#     # return matrix[tf.range(n),idx]
#     return tf.gather_nd(matrix, tf.transpose(tf.stack([tf.range(n), idx])))

def selection_slice(matrix, idx):
    return tf.reduce_sum(matrix * tf.one_hot(idx, matrix.get_shape()[1]), axis=1)

def periodic_saver(variables, name, period):
    saver = tf.train.Saver(variables)
    def save(engine):
        if engine.global_step % period == 0:
            print('Saving {}...'.format(name))
            path = os.path.join(engine.log_dir, name)
            saver.save(get_sess(engine.sess), path, global_step=engine.global_step)
    return save

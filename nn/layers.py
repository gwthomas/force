import numpy as np
import tensorflow as tf

from gtml.util.tf import flatten

def _get_tf(o):
    if isinstance(o, tf.Tensor):
        return o
    elif isinstance(o, Layer):
        return o.get_output()
    else:
        raise RuntimeError

def _get_shape(o):
    return _get_tf(o).get_shape().as_list()


class Layer:
    # input may be either a tf tensor or another Layer instance
    def __init__(self, input, output, params):
        self._input = input
        self._output = output
        self._params = params

    def get_input(self):
        return self._input

    def get_output(self):
        return self._output

    def get_output_shape(self):
        return _get_shape(self._output)

    def get_orig_input(self):
        if isinstance(self._input, tf.Tensor):
            return self._input
        else:
            return self._input.get_orig_input()

    def get_params(self):
        return self._params

    def get_all_params(self):
        if isinstance(self._input, tf.Tensor):
            return self._params
        else:
            return self._input.get_all_params() + self._params

    def eval(self, feed_dict, sess=None):
        if sess is None:
            sess = tf.get_default_session()
        return sess.run(self._output, feed_dict=feed_dict)

    def __call__(self, input):
        return self.eval({self.get_orig_input(): input})


class DenseLayer(Layer):
    def __init__(self, input, size, nonlinearity, num_leading_axes=1):
        input_tf = _get_tf(input)
        flattened_input = flatten(input_tf, num_leading_axes=num_leading_axes)
        input_size = flattened_input.get_shape().as_list()[1]
        self._W = tf.Variable(tf.random_normal([input_size, size], stddev=0.1))
        self._b = tf.Variable(tf.zeros(size))
        pre_activation = tf.matmul(flattened_input, self._W) + self._b
        output = nonlinearity(pre_activation) if nonlinearity is not None else pre_activation
        super().__init__(input, output, [self._W, self._b])


class ConvLayer(Layer):
    def __init__(self, input, num_filters, filter_size, stride, nonlinearity, padding='SAME'):
        input_tf = _get_tf(input)
        input_shape = _get_shape(input_tf)
        in_channels = input_shape[-1]
        out_channels = num_filters
        self._W = tf.Variable(tf.random_normal([filter_size, filter_size, in_channels, out_channels], stddev=0.1))
        self._b = tf.Variable(tf.zeros(out_channels))
        self._params = [self._W, self._b]
        pre_activation = tf.nn.conv2d(input_tf, self._W,
                strides=[1, stride, stride, 1], padding=padding) + self._b
        output = nonlinearity(pre_activation) if nonlinearity is not None else pre_activation
        super().__init__(input, output, [self._W, self._b])


class MaxPoolLayer(Layer):
    def __init__(self, input, size, stride=None, padding='SAME'):
        if stride is None:
            stride = size
        input_tf = _get_tf(input)
        output = tf.nn.max_pool(input_tf, ksize=[1, size, size, 1],
                        strides=[1, stride, stride, 1], padding=padding)
        super().__init__(input, output, [])

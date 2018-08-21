import tensorflow as tf

from gtml.defaults import BATCHSIZE, OPTIMIZER
from gtml.train.supervised import SupervisedLearning


class NoisySupervisedLearning(SupervisedLearning):
    def __init__(self, loss, x_in, y_in, scale, optimizer=OPTIMIZER, batchsize=BATCHSIZE, max_iters=-1):
        self.scale = scale
        SupervisedLearning.__init__(self, loss, x_in, y_in, optimizer=optimizer,
                batchsize=batchsize, max_iters=max_iters)

    def process_grads(self, grads):
        return [(grad + tf.random_normal(grad.shape, stddev=self.scale), var) for grad, var in grads]

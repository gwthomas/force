import numpy as np
import tensorflow as tf

from gtml.common.tf import get_sess
from gtml.defaults import OPTIMIZER
from gtml.train.minimizer import Minimizer


class SupervisedLearning(Minimizer):
    def __init__(self, loss, x_in, y_in, optimizer=OPTIMIZER):
        Minimizer.__init__(self, loss, optimizer=optimizer)
        self.x_in = x_in
        self.y_in = y_in

    def run(self, X, Y, epochs, sess=None):
        for epoch in range(epochs):
            self.epoch({self.x_in: X, self.y_in: Y}, sess=sess)

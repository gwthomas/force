import numpy as np
import tensorflow as tf

from gtml.common.callbacks import CallbackManager
from gtml.common.tf import get_sess
from gtml.defaults import BATCHSIZE, OPTIMIZER


class Minimizer(CallbackManager):
    def __init__(self, loss, optimizer=OPTIMIZER, batchsize=BATCHSIZE):
        CallbackManager.__init__(self)
        self._loss = loss
        self._optimizer = optimizer
        self.batchsize = batchsize
        self._grads = optimizer.compute_gradients(loss)
        self._processed_grads = self.process_grads(self._grads)
        self._apply = optimizer.apply_gradients(self._processed_grads)

        self.epochs_taken = 0
        self.steps_taken = 0
        self.losses = []

    def process_grads(self, grads):
        return grads

    def populate_feed_dict(self, feed_dict):
        pass

    def step(self, feed_dict, sess=None, record=True):
        sess = get_sess(sess)
        self.populate_feed_dict(feed_dict)
        loss, _ = sess.run([self._loss, self._apply], feed_dict=feed_dict)
        if record:
            self.losses.append(loss)
        self.run_callbacks('post-step', sess=sess)
        self.steps_taken += 1
        return loss

    # feeds should be a dictionary mapping placeholders to their data sources
    def epoch(self, feeds, sess=None):
        sess = get_sess(sess)
        n = len(feeds[list(feeds.keys())[0]])
        for ph in feeds:
            assert n == len(feeds[ph])

        shuffled_indices = np.random.permutation(n)
        steps = int(np.ceil(float(n) / self.batchsize))
        losses = []
        for step in range(steps):
            start = step * self.batchsize
            end = min(n, (step + 1) * self.batchsize)
            idx = shuffled_indices[start:end]
            loss = self.step({ph: feeds[ph][idx] for ph in feeds}, sess=sess)
            losses.append(loss)
        self.run_callbacks('post-epoch', sess=sess)
        self.epochs_taken += 1
        return losses

import tensorflow as tf

from force.callbacks import CallbackManager
from force.constants import DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS, INF
from force.serialization import Serializable


class Minimizer(CallbackManager, Serializable):
    def __init__(self, loss_fn, parameters, optimizer):
        CallbackManager.__init__(self)
        self.loss_fn = loss_fn
        self.parameters = parameters
        self.steps_taken = 0

        if isinstance(optimizer, tf.keras.optimizers.Optimizer):
            self.optimizer = optimizer
        elif callable(optimizer):
            self.optimizer = optimizer()
        else:
            raise RuntimeError('Invalid optimizer: {}'.format(optimizer))

    def _state_attrs(self):
        return ['steps_taken']

    def step(self, inputs):
        self.run_callbacks('pre_step', self.steps_taken)
        with tf.GradientTape() as tape:
            loss = self.loss_fn(inputs)
            grad = tape.gradient(loss, self.parameters)
        self.optimizer.apply_gradients(zip(grad, self.parameters))
        self.steps_taken += 1
        loss_val = float(loss)
        self.run_callbacks('post_step', self.steps_taken, loss_val)


class EpochalMinimizer(Minimizer):
    def __init__(self, loss_fn, parameters, optimizer, dataset):
        Minimizer.__init__(self, loss_fn, parameters, optimizer)
        self.dataset = dataset
        self.epochs_taken = 0

    def _state_attrs(self):
        return Minimizer._state_attrs(self) + ['epochs_taken']

    def run(self, n_epochs, max_epochs=INF):
        for _ in range(n_epochs):
            if self.epochs_taken >= max_epochs:
                return
            self.run_callbacks('pre_epoch', self.epochs_taken)
            for batch in self.dataset:
                self.step(batch)
            self.epochs_taken += 1
            self.run_callbacks('post_epoch', self.epochs_taken)

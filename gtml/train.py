import torch
from torch.utils.data import DataLoader

from gtml.callbacks import CallbackManager
from gtml.constants import DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS, DEVICE
from gtml.core import Serializable
import gtml.util as util


class LossFunction:
    def __init__(self):
        self.terms = {}
        
    def add_term(self, name, compute_fn):
        self.terms[name] = compute_fn
        
    def add_supervised_term(self, name, model, criterion):
        def supervised_loss(inputs):
            x, y = inputs['x'], inputs['y']
            return criterion(model(x), y)
        self.add_term(name, supervised_loss)
        
    def __call__(self, inputs):
        total = 0
        terms = {}
        for name, compute_fn in self.terms.items():
            term = compute_fn(inputs)
            total = total + term
            terms[name] = term
        return total, terms


class Minimizer(CallbackManager, Serializable):
    def __init__(self, loss_fn, optimizer):
        CallbackManager.__init__(self)
        self.loss_fn = loss_fn
        self.steps_taken = 0

        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        elif callable(optimizer):
            self.optimizer = optimizer()
        else:
            raise RuntimeError('Invalid optimizer: {}'.format(optimizer))

    def _state_attrs(self):
        return ['steps_taken']

    def step(self, inputs):
        self.run_callbacks('pre_step', self.steps_taken)
        self.optimizer.zero_grad()
        loss, terms = self.loss_fn(inputs)
        loss.backward()
        self.optimizer.step()
        self.steps_taken += 1
        loss_val = util.scalar(loss)
        term_vals = {name: util.scalar(term) for name, term in terms.items()}
        self.run_callbacks('post_step', self.steps_taken, loss_val, term_vals)


class EpochalMinimizer(Minimizer):
    def __init__(self, loss_fn, optimizer, data_loader=None):
        Minimizer.__init__(self, loss_fn, optimizer)
        self.epochs_taken = 0
        self.data_loader = data_loader

    def create_data_loader(self, dataset, batch_size=DEFAULT_BATCH_SIZE,
                           num_workers=DEFAULT_NUM_WORKERS):
        self.data_loader = DataLoader(dataset, batch_size=batch_size,
                                      shuffle=True, pin_memory=True,
                                      num_workers=num_workers)

    def _state_attrs(self):
        return Minimizer._state_attrs(self) + ['epochs_taken']

    def run(self, max_epochs):
        if self.data_loader is None:
            raise RuntimeError('EpochalMinimizer has no data loader')

        while self.epochs_taken < max_epochs:
            self.run_callbacks('pre_epoch', self.epochs_taken)
            for x, y in self.data_loader:
                self.step({'x': x.to(DEVICE), 'y': y.to(DEVICE)})
            self.epochs_taken += 1
            self.run_callbacks('post_epoch', self.epochs_taken)

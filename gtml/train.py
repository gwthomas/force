import torch
from torch.utils.data import DataLoader

from gtml.callbacks import CallbackManager
from gtml.constants import DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS, DEVICE
from gtml.core import Serializable


class Minimizer(CallbackManager, Serializable):
    def __init__(self, compute_loss, optimizer):
        CallbackManager.__init__(self)
        self.compute_loss = compute_loss
        self.steps_taken = 0

        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        elif callable(optimizer):
            self.optimizer = optimizer()
        else:
            raise RuntimeError('Invalid optimizer: {}'.format(optimizer))

    def _state_attrs(self):
        return ['steps_taken']

    def step(self, *inputs):
        self.run_callbacks('pre_step', self.steps_taken)
        self.optimizer.zero_grad()
        loss = self.compute_loss(*inputs)
        loss.backward()
        lossval = loss.item()
        self.optimizer.step()
        self.steps_taken += 1
        self.run_callbacks('post_step', self.steps_taken, lossval)


class EpochalMinimizer(Minimizer):
    def __init__(self, compute_loss, optimizer, dataset,
                 batch_size=DEFAULT_BATCH_SIZE,
                 num_workers=DEFAULT_NUM_WORKERS):
        Minimizer.__init__(self, compute_loss, optimizer)
        self.batch_size = batch_size
        self.epochs_taken = 0
        self.data_loader = DataLoader(dataset, batch_size=batch_size,
                                      shuffle=True, pin_memory=True,
                                      num_workers=num_workers)

    def _state_attrs(self):
        return Minimizer._state_attrs(self) + ['epochs_taken']

    def run(self, max_epochs):
        while self.epochs_taken < max_epochs:
            self.run_callbacks('pre_epoch', self.epochs_taken)
            for batch in self.data_loader:
                batch = [item.to(DEVICE) for item in batch]
                self.step(*batch)
            self.epochs_taken += 1
            self.run_callbacks('post_epoch', self.epochs_taken)

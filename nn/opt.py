import torch; nn = torch.nn
from math import ceil

from gtml.common.callbacks import CallbackManager
from gtml.defaults import OPTIMIZER, BATCHSIZE


class Minimizer(CallbackManager):
    def __init__(self, params, compute_loss, optimizer=None, batchsize=BATCHSIZE):
        CallbackManager.__init__(self)
        self.compute_loss = compute_loss
        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        elif optimizer is None:
            optimizer = OPTIMIZER
        self.optimizer = optimizer(params)
        self.batchsize = batchsize
        self.epochs_taken = 0
        self.steps_taken = 0
        self.record_loss = True
        self.losses = []

    def step(self, inputs):
        self.optimizer.zero_grad()
        loss = self.compute_loss(*inputs)
        loss.backward()
        self.optimizer.step()
        lossval = float(loss.data)
        if self.record_loss:
            self.losses.append(lossval)
        self.run_callbacks('post-step')
        self.steps_taken += 1
        return lossval

    def epoch(self, inputs):
        n = len(inputs[0])
        for input in inputs:
            assert n == len(input)

        shuffled_indices = torch.randperm(n)
        steps = ceil(float(n) / self.batchsize)
        losses = []
        for step in range(steps):
            start = step * self.batchsize
            end = min(n, (step + 1) * self.batchsize)
            idx = shuffled_indices[start:end]
            loss = self.step([input[idx] for input in inputs])
            losses.append(loss)
        self.run_callbacks('post-epoch')
        self.epochs_taken += 1
        return losses


class SupervisedLearning(Minimizer):
    def __init__(self, net, criterion, optimizer=None, batchsize=BATCHSIZE):
        def compute_loss(input, target):
            output = net(input)
            return criterion(output, target)
        Minimizer.__init__(self, net.parameters(), compute_loss,
                optimizer=optimizer, batchsize=batchsize)

    def run(self, X, Y, epochs):
        for epoch in range(epochs):
            self.epoch([X, Y])

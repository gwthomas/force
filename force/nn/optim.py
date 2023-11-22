import torch

from force import defaults
from force.nn.module import ConfigurableModule


class Optimizer(ConfigurableModule):
    class Config(ConfigurableModule.Config):
        algorithm = defaults.OPTIMIZER_TYPE
        lr = defaults.LEARNING_RATE
        kwargs = {} # optimizer-specific arguments

    def __init__(self, cfg, parameters):
        super().__init__(cfg)
        optimizer_class = getattr(torch.optim, cfg.algorithm)
        assert 'lr' not in cfg.kwargs
        kwargs = dict(cfg.kwargs, lr=cfg.lr)
        self.optimizer = optimizer_class(parameters, **kwargs)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def __repr__(self):
        return f'{self.cfg.algorithm}(lr={self.cfg.lr})'
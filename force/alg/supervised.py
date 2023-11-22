import torch
from torch.utils.data import DataLoader

from force.nn.optim import Optimizer

from .base import IterativeAlgorithm


class SupervisedLearning(IterativeAlgorithm):
    class Config(IterativeAlgorithm.Config):
        max_iterations = 1000000
        batch_size = 32
        num_workers = 1
        eval_period = 1000
        optimizer = Optimizer.Config()

    def __init__(self, cfg, model, loss_fn, train_set, eval_set,
                 extra_eval_metrics=None):
        super().__init__(cfg)
        self.model = model
        self.loss_fn = loss_fn
        self.train_set = train_set
        self.eval_set = eval_set
        self.extra_eval_metrics = extra_eval_metrics
        if extra_eval_metrics is not None:
            assert isinstance(extra_eval_metrics, dict)

        # Optimizer
        self.optimizer = Optimizer(cfg.optimizer, model.parameters())

        # Loaders
        loader_kwargs = {
            'batch_size': self.cfg.batch_size,
            'num_workers': self.cfg.num_workers,
            'drop_last': True
        }
        self.train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
        self.eval_loader = DataLoader(eval_set, shuffle=False, **loader_kwargs)
        self.train_iterator = None
        self.epochs_done = 0

    def get_counters(self) -> dict:
        return {
            **super().get_counters(),
            'epochs': self.epochs_done
        }

    def get_batch(self):
        try:
            return next(self.train_iterator)
            # exception if self.train_iterator is None (as initialized)
            # or next raises StopIteration
        except:
            if self.train_iterator is not None:
                self.epochs_done += 1
            self.log('Starting new epoch!')
            self.train_iterator = iter(self.train_loader)
            return self.get_batch()

    def iteration(self):
        x, y = self.get_batch()
        outputs = self.model(x)
        loss = self.loss_fn(outputs, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            'loss': loss.item()
        }

    def post_iteration(self):
        if self.iterations_done % self.eval_period == 0:
            return self.evaluate()
        else:
            return {}

    def evaluate(self) -> dict:
        metrics = {'loss': lambda o, y: self.loss_fn(o, y).item()}
        if self.extra_eval_metrics:
            metrics.update(self.extra_eval_metrics)
        metric_totals = {key: 0. for key in metrics.keys()}
        total_samples = 0
        eval_iterator = iter(self.eval_loader)

        self.model.eval()
        for x, y in eval_iterator:
            batch_size = len(x)
            total_samples += batch_size
            outputs = self.model(x)
            with torch.no_grad():
                for key, metric in metrics.items():
                    metric_totals[key] += batch_size * metric(outputs, y)

        self.model.train()

        return {
            f'eval/{key}': metric_totals[key] / total_samples
            for key in metrics.keys()
        }


def accuracy(outputs, y):
    # Assumes highest output is prediction (e.g. logits or probabilities)
    predictions = outputs.argmax(-1)
    assert predictions.shape == y.shape
    return (predictions == y).to(float).mean()

class Classification(SupervisedLearning):
    def  __init__(self, cfg, model, train_set, eval_set):
        loss_fn = torch.nn.CrossEntropyLoss()
        extra_eval_metrics = {'accuracy': accuracy}
        super().__init__(cfg, model, loss_fn,
                         train_set, eval_set,
                         extra_eval_metrics)


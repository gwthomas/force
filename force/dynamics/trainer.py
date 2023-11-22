from force import defaults
from force.nn.module import ConfigurableModule
from force.nn.optim import Optimizer
from force.nn.util import ModuleWrapper


class DynamicsModelTrainer(ConfigurableModule):
    class Config(ConfigurableModule.Config):
        optimizer = Optimizer.Config()
        batch_size = defaults.BATCH_SIZE

    def __init__(self, cfg, model):
        super().__init__(cfg)
        self.optimizer = Optimizer(cfg.optimizer, model.parameters())
        self.model = ModuleWrapper(model)

        # Special handling for ensembles
        self.num_models = model.num_models if hasattr(model, 'num_models') else 1

    def update(self, buffer):
        # Sample and reshape batch
        batch = buffer.sample(self.num_models * self.cfg.batch_size)

        # Special handling for ensembles
        if self.num_models > 1:
            batch_dims = (self.cfg.batch_size, self.num_models)
            for k in batch.keys():
                v = batch[k]
                if v.ndim > 1:  # vectors
                    batch[k] = v.reshape(*batch_dims, -1)
                else:           # scalars
                    batch[k] = v.reshape(*batch_dims)

        # Model forward
        distributions = self.model.distribution(batch['observations'], batch['actions'])

        # Optimize NLL loss
        total_log_prob = distributions['next_obs'].log_prob(batch['next_observations']) + \
                         distributions['reward'].log_prob(batch['rewards']) + \
                         distributions['terminal'].log_prob(batch['terminals'].float())
        loss = -total_log_prob.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
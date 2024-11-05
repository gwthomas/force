import torch

from force.nn.module import Module
from force.nn.util import get_device
from force.policies import NeuralPolicy


class Normalizer(Module):
    def __init__(self, shape, device=None):
        super().__init__()
        assert isinstance(shape, torch.Size) and len(shape) == 1
        self.shape = shape
        self.dim = dim = shape[0]
        self._device = device = get_device(device)
        self.register_buffer('mean', torch.zeros(dim, device=device))
        self.register_buffer('var', torch.ones(dim, device=device))
        self.register_buffer('count', torch.zeros([], dtype=int, device=device))

    def fit(self, x):
        assert x.ndim == 2
        assert x.shape[1] == self.dim
        var, mean = torch.var_mean(x, dim=0)
        self.mean.copy_(mean)
        self.var.copy_(var)
        self.count.fill_(len(x))

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_var, batch_mean = torch.var_mean(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def get_output_shape(self, input_shape):
        assert input_shape[-1] == self.dim
        return input_shape

    def forward(self, x):
        std = torch.sqrt(self.var)
        return (x - self.mean) / std

    def extra_repr(self):
        return f'dim={self.dim}'


class InputNormalizerWrapper(Module):
    def __init__(self, module, normalizer):
        super().__init__()
        self.module = module
        self.normalizer = normalizer

    def get_output_shape(self, input_shape, **kwargs):
        return self.module.get_output_shape(input_shape, **kwargs)

    def forward(self, x):
        x = self.normalizer(x)
        return self.module(x)


# Special case
class InputNormalizedPolicy(InputNormalizerWrapper):
    def __init__(self, policy, normalizer):
        assert isinstance(policy, NeuralPolicy)
        super().__init__(policy, normalizer)

    def distribution(self, obs):
        obs = self.normalizer(obs)
        return self.module.distribution(obs)

    def act(self, obs, eval: bool):
        obs = self.normalizer(obs)
        return self.module.act(obs, eval)
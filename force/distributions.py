import math
import torch.nn.functional as F
import torch.distributions as D


class DiagonalGaussian(D.Distribution):
    def __init__(self, loc, scale):
        assert loc.shape == scale.shape
        assert (scale > 0).all()
        self.loc = loc
        self.scale = scale
        self._batch_shape = loc.shape[:-1]
        self._event_shape = loc.shape[-1:]

        self._impl = D.Independent(D.Normal(loc, scale), 1)
        assert self._impl.batch_shape == self._batch_shape
        assert self._impl.event_shape == self._event_shape

        for method in ['sample', 'rsample', 'log_prob']:
            setattr(self, method, getattr(self._impl, method))


# Borrowed from https://github.com/denisyarats/pytorch_sac

class TanhTransform(D.transforms.Transform):
    domain = D.constraints.real
    codomain = D.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedGaussian(D.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        super().__init__(DiagonalGaussian(loc, scale), TanhTransform())

    @property
    def mean(self):
        mu = self.base_dist.loc
        for transform in self.transforms:
            mu = transform(mu)
        return mu
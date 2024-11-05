import torch.distributions as D


class SquashedDistribution(D.transformed_distribution.TransformedDistribution):
    def __init__(self, base_dist):
        self.tanh_transform = D.transforms.TanhTransform(cache_size=1)
        super().__init__(base_dist, self.tanh_transform)

    @property
    def mode(self):
        return self.tanh_transform(self.base_dist.mode)


def diagonal_gaussian(loc, scale, reinterpreted_batch_ndims=1, squash=False):
    distr = D.Normal(loc, scale)
    if squash:
        distr = SquashedDistribution(distr)
    if reinterpreted_batch_ndims > 0:
        distr = D.Independent(distr, reinterpreted_batch_ndims)
    return distr
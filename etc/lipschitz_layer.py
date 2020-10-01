import torch
import torch.nn as nn
import torch.nn.functional as F

from force.torch_util import device


class SpectralNormalizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, power_iters=1):
        super().__init__(in_features, out_features, bias)
        self.sigma = None
        self.power_iters = power_iters
        self.register_buffer('u', torch.randn(self.weight.shape[1]))
        self.one = torch.tensor(1.).to(device)

    def train(self, mode=True):
        sigma = torch.svd(self.weight, compute_uv=False).S.max().detach().cpu()
        if mode:
            print('Performing initial power iteration')
            sigma_hat = torch.tensor(0.)
            while not torch.isclose(sigma_hat, sigma):
                sigma_hat = self.power_iteration().detach().cpu()
            self.sigma = None
        else:
            self.sigma = sigma

    def power_iteration(self):
        A = self.weight
        u_hat, v_hat = self.u, None
        with torch.no_grad():
            for _ in range(self.power_iters):
                v_ = u_hat @ A.t()
                v_hat = v_ / torch.norm(v_)
                u_ = v_hat @ A
                u_hat = u_ / torch.norm(u_)
        sigma = v_hat @ A @ u_hat
        self.u.set_(u_hat)
        return sigma

    def forward(self, input):
        sigma = self.power_iteration() if self.training else self.sigma
        denom = torch.max(sigma, self.one)
        return F.linear(input, self.weight / denom, self.bias)
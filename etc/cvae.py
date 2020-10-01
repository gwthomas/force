import torch
import torch.nn as nn
import torch.nn.functional as F

from force.torch_util import device, Module, mlp


# Adapted from https://github.com/sfujim/BCQ


class CVAE(Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, hidden_dim=750):
        super(CVAE, self).__init__()
        self.encoder = mlp([state_dim + action_dim, hidden_dim, hidden_dim])

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.decoder = mlp([state_dim + latent_dim, hidden_dim, hidden_dim, action_dim])

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action):
        z = F.relu(self.encoder(torch.cat([state, action], 1)))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        # z = mean + std * torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(device)
        z = mean + std * torch.normal(0, 1, size=std.size()).to(device)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            # z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.5, 0.5)
            z = torch.normal(0, 1, size=(state.size(0), self.latent_dim)).to(device).clamp(-0.5, 0.5)

        return self.max_action * torch.tanh(self.decoder(torch.cat([state, z], 1)))

    def loss(self, state, action):
        recon, mean, std = self(state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        return recon_loss + 0.5 * KL_loss
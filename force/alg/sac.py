import torch
import torch.nn as nn

from force.alg.actor_critic import BufferedActorCritic
from force.config import Field, Choice
from force.nn import Optimizer
from force.nn.util import get_device, torchify, batch_map, update_ema
from force.nn.shape import shape_numel
from force.policies import GaussianPolicy


class SAC(BufferedActorCritic):
    class Config(BufferedActorCritic.Config):
        actor = GaussianPolicy.Config
        init_alpha = 1.0
        tune_alpha = True
        target_entropy = Field(float, required=False)
        use_log_alpha_loss = True
        deterministic_backup = False

    def __init__(self, cfg, env_info,
                 actor=None, device=None):
        if actor is None:
            actor = GaussianPolicy(cfg.actor, env_info)

        super().__init__(cfg, env_info, actor,
                         use_target_actor=False, use_target_critic=True,
                         device=device)

        self.log_alpha = torch.tensor(cfg.init_alpha, device=self.device).log()
        if self.cfg.tune_alpha:
            self.log_alpha = nn.Parameter(self.log_alpha)
            self.alpha_optimizer = Optimizer(cfg.actor_optimizer, [self.log_alpha])
            if self.cfg.target_entropy is None:
                self.cfg.target_entropy = -shape_numel(act_shape)   # set target entropy to -dim(A)

        # Check if policy can compute exact entropy
        try:
            rand_obs = torchify(env_info.observation_space.sample()).unsqueeze(0)
            with torch.no_grad():
                self.actor.distribution(rand_obs).entropy()
            self.exact_entropy = True
        except NotImplementedError:
            self.exact_entropy = False

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def compute_target_value(self, obs):
        distr = self.actor.distribution(obs)
        action = distr.rsample()
        value = self.target_critic([obs, action], which='min')
        if not self.cfg.deterministic_backup:
            if self.exact_entropy:
                entropy = distr.entropy()
            else:
                entropy = -distr.log_prob(action)
            value = value + self.alpha * entropy
        return value

    def compute_actor_loss(self, obs):
        cfg = self.cfg
        distr = self.actor.distribution(obs)
        action = distr.rsample()
        if self.exact_entropy:
            entropy = distr.entropy()
        else:
            entropy = -distr.log_prob(action)
        actor_q = self.critic([obs, action], which='min')
        alpha = self.alpha
        actor_loss = -torch.mean(actor_q + alpha.detach() * entropy)

        if cfg.tune_alpha:
            multiplier = self.log_alpha if self.cfg.use_log_alpha_loss else alpha
            alpha_loss = multiplier * torch.mean(entropy.detach() - self.cfg.target_entropy)
            return [actor_loss, alpha_loss]
        else:
            return [actor_loss]

    def update_actor(self, obs):
        losses = self.compute_actor_loss(obs)
        self.train_diagnostics['actor_loss'].append(losses[0].item())
        optimizers = [self.actor_optimizer]
        if self.cfg.tune_alpha:
            optimizers.append(self.alpha_optimizer)
        assert len(losses) == len(optimizers)
        for loss, optimizer in zip(losses, optimizers):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def additional_diagnostics(self) -> dict:
        info = super().additional_diagnostics()
        info['alpha'] = self.alpha.detach()

        obs = self.replay_buffer.get('observations')
        def entropy(o):
            distr = self.actor.distribution(o)
            if self.exact_entropy:
                return distr.entropy()
            else:
                return -distr.log_prob(distr.sample())
        with torch.no_grad():
            info['entropy'] = batch_map(entropy, obs).mean()
        return info
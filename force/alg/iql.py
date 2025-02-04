import torch
import torch.nn.functional as F

from force.alg.actor_critic import BufferedActorCritic
from force.defaults import NUMERICAL_EPSILON
from force.nn import Optimizer
from force.nn.models.value_functions import ValueFunction
from force.nn.util import get_device
from force.policies import GaussianPolicy
from force.util import dict_get, pymean


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class IQL(BufferedActorCritic):
    class Config(BufferedActorCritic.Config):
        beta = 3.0
        tau = 0.7
        exp_adv_max = 100.0
        actor = GaussianPolicy.Config
        vf = ValueFunction.Config
        vf_optimizer = Optimizer.Config

    def __init__(self, cfg, env_info, actor=None, device=None):
        if actor is None:
            actor = GaussianPolicy(cfg.actor, env_info)
        device = get_device(device)
        super().__init__(cfg, env_info, actor,
                         use_target_actor=False, use_target_critic=True,
                         device=device)

        self.vf = ValueFunction(cfg.vf, env_info.action_shape).to(self.device)
        self.vf_optimizer = Optimizer(cfg.vf_optimizer, self.vf.parameters())

    def compute_target_value(self, obs):
        return self.vf(obs)

    def update_with_batch(self, batch: dict, counters: dict):
        obs, actions, next_obs, rewards, terminals, _ = dict_get(batch, *batch.keys())

        with torch.no_grad():
            target_q = self.target_critic([obs, actions], which='min')
            next_v = self.vf(next_obs)

        # Value function
        v = self.vf(obs)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.cfg.tau)
        self.train_diagnostics['v_loss'].append(v_loss.item())
        self.vf_optimizer.zero_grad()
        v_loss.backward()
        self.vf_optimizer.step()

        # Q function
        targets = rewards + (~terminals).float() * self.cfg.discount * next_v
        qs = self.critic([obs, actions], which='all')
        q_loss = pymean([F.mse_loss(qs[:,i], targets) for i in range(self.critic.num_models)])
        self.train_diagnostics['q_loss'].append(q_loss.item())
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # Actor
        exp_adv = torch.exp(self.cfg.beta * adv.detach()).clamp(max=self.cfg.exp_adv_max)
        act_distr = self.actor.distribution(obs)
        log_probs = act_distr.log_prob(
            actions.clamp(  # for stability
                min=-1+NUMERICAL_EPSILON,
                max= 1-NUMERICAL_EPSILON
            )
        )
        actor_loss = -torch.mean(exp_adv * log_probs)
        self.train_diagnostics['actor_loss'].append(actor_loss.item())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Target network
        self.update_target_networks()
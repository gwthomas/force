import torch
import torch.nn.functional as F

from force.alg.actor_critic import ActorCritic
from force.config import Configurable, Field
from force.env.util import space_dim, space_shape
from force.nn import MLP, Optimizer
from force.policies import GaussianPolicy
from force.util import dict_get_several, pymean
from force.value_functions import QFunctionEnsemble, ValueFunction


ACTION_EPSILON = 1e-6
EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class IQL(ActorCritic):
    class Config(ActorCritic.Config):
        beta = 3.0
        tau = 0.7
        actor = GaussianPolicy.Config()
        critic = QFunctionEnsemble.Config()
        vf = ValueFunction.Config()
        vf_optimizer = Optimizer.Config()

    def __init__(self, cfg, obs_space, act_space,
                 actor=None, critic=None):
        Configurable.__init__(self, cfg)
        obs_shape = space_shape(obs_space)
        act_shape = space_shape(act_space)
        act_dim = act_shape[0]
        if actor is None:
            actor = GaussianPolicy(self.cfg.actor, obs_shape, act_dim)
        if critic is None:
            critic = QFunctionEnsemble(self.cfg.critic, obs_shape, act_shape)
        super().__init__(cfg, obs_space, act_space, actor, critic,
                         use_actor_target=False, use_critic_target=True)

        self.vf = ValueFunction(cfg.vf, obs_shape)
        self.vf_optimizer = Optimizer(cfg.vf_optimizer, self.vf.parameters())

    def update_with_batch(self, batch: dict, counters: dict):
        obs, actions, next_obs, rewards, terminals, _ = dict_get_several(batch, *batch.keys())

        with torch.no_grad():
            target_q = self.critic_target([obs, actions], which='min')
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
        targets = rewards + (~terminals).float() * self.cfg.discount * next_v.detach()
        qs = self.critic([obs, actions], which='all')
        q_loss = pymean([F.mse_loss(qs[:,i], targets) for i in range(self.critic.num_models)])
        self.train_diagnostics['q_loss'].append(q_loss.item())
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # Actor
        exp_adv = torch.exp(self.cfg.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        act_distr = self.actor.distr(obs)
        log_probs = act_distr.log_prob(
            actions.clamp(  # for stability
                min=-1+ACTION_EPSILON,
                max= 1-ACTION_EPSILON
            )
        )
        actor_loss = -torch.mean(exp_adv * log_probs)
        self.train_diagnostics['actor_loss'].append(actor_loss.item())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Target network
        self.update_target_networks()
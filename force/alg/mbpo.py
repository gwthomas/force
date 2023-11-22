import torch
import torch.nn.functional as F
from tqdm import trange

from force.alg.base import Agent
from force.alg.sac import SAC
from force.config import Field
from force.dynamics import GaussianDynamicsEnsemble, DynamicsModelTrainer
from force.nn.util import get_device
from force.sampling import ReplayBuffer
from force.util import batch_map, prefix_dict_keys, pymean


class MBPO(Agent):
    class Config(Agent.Config):
        solver = SAC.Config()
        solver_updates = 20
        model = GaussianDynamicsEnsemble.Config()
        model_trainer = DynamicsModelTrainer.Config()
        model_update_period = 250
        model_updates = 1000
        initial_model_updates = 10000
        horizon = 1
        rollout_batch_size = 400
        model_buffer_capacity = 100000
        real_frac = Field(0.5, check=lambda x: 0 <= x <= 1)

    def __init__(self, cfg, obs_space, act_space, device=None):
        super().__init__(cfg, obs_space, act_space)
        device = get_device(device)

        self.solver = SAC(cfg.solver, obs_space, act_space, device=device)
        self.model_ensemble = GaussianDynamicsEnsemble(cfg.model, obs_space, act_space).to(device)
        self.model_trainer = DynamicsModelTrainer(cfg.model_trainer, self.model_ensemble)
        self.model_buffer = ReplayBuffer(obs_space, act_space, cfg.model_buffer_capacity, device=device)

        self._initial_fit_done = False

    def act(self, obs, eval):
        return self.solver.act(obs, eval)

    def _update_models(self, data, num_updates):
        self.log('Updating models...')
        losses = [self.model_trainer.update(data) for _ in trange(num_updates)]

        N = 10  # number of losses to average over when logging
        self.log(f'Loss went from {pymean(losses[:N])} to {pymean(losses[-N:])}')

    def _mb_rollouts(self, initial_states):
        states = initial_states
        for t in range(self.cfg.horizon):
            with torch.no_grad():
                actions = self.solver.act(states, eval=False)
                next_states, rewards, terminals = self.model_ensemble.sample(states, actions)
            truncateds = torch.full_like(terminals, t == self.cfg.horizon - 1)
            self.model_buffer.extend({
                'observations': states,
                'actions': actions,
                'next_observations': next_states,
                'rewards': rewards,
                'terminals': terminals,
                'truncateds': truncateds
            })

            if terminals.all():
                break
            else:
                states = next_states[~terminals]

    def _get_mixed_batch(self, data):
        batch_size = self.solver.cfg.batch_size
        num_real = int(self.cfg.real_frac * batch_size)
        real_batch = data.sample(num_real)
        model_batch = self.model_buffer.sample(batch_size - num_real)
        return {
            k: torch.cat([real_batch[k], model_batch[k]])
            for k in real_batch.keys()
        }

    def update(self, data, counters):
        cfg = self.cfg
        if not self._initial_fit_done:
            self._update_models(data, cfg.initial_model_updates)
            self._initial_fit_done = True
        elif counters['env_steps'] % cfg.model_update_period == 0:
            self._update_models(data, cfg.model_updates)

        # Generate rollouts
        initial_states = data.sample(cfg.rollout_batch_size)['observations']
        self._mb_rollouts(initial_states)

        # Update agent
        for _ in range(cfg.solver_updates):
            batch = self._get_mixed_batch(data)
            self.solver.update_with_batch(batch, counters)

    def additional_diagnostics(self, data: ReplayBuffer) -> dict:
        diagnostics = super().additional_diagnostics(data)
        solver_diagnostics = self.solver.additional_diagnostics(data)
        diagnostics.update(prefix_dict_keys('solver', solver_diagnostics))

        data = data.get(as_dict=True)
        states = data['observations']
        actions = data['actions']
        next_states = data['next_observations']
        rewards = data['rewards']
        terminals = data['terminals']
        with torch.no_grad():
            ns_pred, r_pred, t_pred = batch_map(self.model_ensemble.mean, (states, actions))
        assert next_states.shape == ns_pred.shape
        assert rewards.shape == r_pred.shape
        diagnostics['next_state_error'] = F.mse_loss(ns_pred, next_states)
        diagnostics['reward_error'] = F.mse_loss(r_pred, rewards)
        diagnostics['terminal_error'] = F.mse_loss(t_pred, terminals)
        # diagnostics['terminal_error'] = (t_pred == terminals).float().mean()
        return diagnostics
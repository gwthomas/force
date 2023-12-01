import itertools
import random

import torch

from force.alg.base import Agent
from force.alg.sac import SAC
from force.config import Field
from force.dynamics import GaussianDynamicsEnsemble
from force.nn.layers import LinearEnsemble
from force.nn.util import get_device, freepeat
from force.sampling import ReplayBuffer
from force.util import batch_map, prefix_dict_keys, pymean


# Batch size used for computing next states and evaluations.
# Does not affect training. Adjust according to CUDA memory limits.
BATCH_MAP_SIZE = 10000


class MBPO(Agent):
    class Config(Agent.Config):
        solver = SAC
        solver_updates = 20
        model = GaussianDynamicsEnsemble
        num_elites = 5
        epoch_length = 1000
        model_train_period = 250
        max_epochs_since_update = 5
        rollout_schedule = [20, 100, 1, 1]
        rollout_batch_size = 100000
        model_retain_epochs = 1
        real_frac = Field(0.1, check=lambda x: 0 <= x <= 1)

    def __init__(self, cfg, obs_space, act_space, device=None,
                 termination_fn=None):
        assert cfg.num_elites <= cfg.model.ensemble.num_models
        super().__init__(cfg, obs_space, act_space)
        device = get_device(device)

        self.solver = SAC(cfg.solver, obs_space, act_space, device=device)
        self.model_ensemble = GaussianDynamicsEnsemble(
            cfg.model, obs_space, act_space, termination_fn=termination_fn
        ).to(device)
        self._ensemble_layers = [
            m for m in self.model_ensemble.modules() if isinstance(m, LinearEnsemble)
        ]
        self._best_parameters = [
            None for _ in range(self.model_ensemble.num_models)
        ]
        self._elite_indices = None
        self.model_buffer = None

    def act(self, obs, eval):
        return self.solver.act(obs, eval)

    def _save_parameters(self, index):
        params = []
        for m in self._ensemble_layers:
            params.append((
                m.weight[index].clone().detach(),
                m.bias[index].clone().detach()
            ))
        self._best_parameters[index] = params

    def _load_best_parameters(self):
        for i in range(self.model_ensemble.num_models):
            assert self._best_parameters[i] is not None

        for m, params in zip(self._ensemble_layers, zip(*self._best_parameters)):
            for i in range(self.model_ensemble.num_models):
                m.weight.data[i], m.bias.data[i] = params[i]

    def _train_models(self, data):
        tensors = data.get()[:-1] # ignore truncated

        # Repeat
        num_models = self.model_ensemble.num_models
        tensors_repeated = [freepeat(x, num_models, dim=1) for x in tensors]

        def compute_ll():
            with torch.no_grad():
                return batch_map(
                    self.model_ensemble.log_likelihood,
                    tensors_repeated,
                    batch_size=BATCH_MAP_SIZE
                ).mean(0).cpu()

        epochs_since_update = 0
        best_lls = compute_ll()
        for epoch in itertools.count():
            self.model_ensemble.epoch(*tensors)
            lls = compute_ll()

            updated = False
            for i in range(num_models):
                if lls[i] > best_lls[i]:
                    best_lls[i] = lls[i]
                    updated = True
                    self._save_parameters(i)

            if epoch % 10 == 0:
                self.log(f'Model epoch {epoch}: {lls.mean().item():.2f} (best: {best_lls.mean().item():.2f})')

            if updated:
                epochs_since_update = 0
            else:
                epochs_since_update += 1

            if epochs_since_update > self.cfg.max_epochs_since_update:
                self._load_best_parameters()

                # Check we reproduce best lls
                final_lls = compute_ll()
                self.log(f'Final: {final_lls}')
                self.log(f'Best : {best_lls}')

                elite_info = torch.topk(best_lls, self.cfg.num_elites)
                self._elite_indices = elite_info.indices.tolist()
                self._model_log_likelihood = lls.mean()
                self._elite_log_likelihood = elite_info.values.mean()
                return

    def _set_rollout_length(self, epoch):
        min_epoch, max_epoch, min_length, max_length = self.cfg.rollout_schedule
        if epoch <= min_epoch:
            y = min_length
        else:
            dx = (epoch - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_length - min_length) + min_length

        self._rollout_length = int(y)
        self.log(f'Rollout length: {self._rollout_length}')

    def _reallocate_model_buffer(self):
        cfg = self.cfg
        rollouts_per_epoch = cfg.rollout_batch_size * cfg.epoch_length / cfg.model_train_period
        model_steps_per_epoch = int(self._rollout_length * rollouts_per_epoch)
        new_capacity = cfg.model_retain_epochs * model_steps_per_epoch

        if self.model_buffer is None:
            self.log(f'Initializing new model buffer with size {new_capacity}')
            self.model_buffer = ReplayBuffer(self.obs_space, self.act_space, new_capacity)
        elif self.model_buffer.capacity != new_capacity:
            self.log(f'Updating model buffer: {self.model_buffer.capacity} -> {new_capacity}')
            samples = self.model_buffer.get(as_dict=True)
            new_buffer = ReplayBuffer(self.obs_space, self.act_space, new_capacity)
            new_buffer.extend(samples)
            assert len(self.model_buffer) == len(new_buffer)
            self.model_buffer = new_buffer

    def _act_and_transition(self, states):
        actions = self.solver.act(states, eval=False)
        next_states, rewards, terminals = self.model_ensemble.sample(
            states, actions, model_index=random.choice(self._elite_indices)
        )
        return actions, next_states, rewards, terminals

    def _mb_rollouts(self, initial_states):
        states = initial_states
        steps_added = 0
        for t in range(self._rollout_length):
            with torch.no_grad():
                actions, next_states, rewards, terminals = batch_map(
                    self._act_and_transition, states,
                    batch_size=BATCH_MAP_SIZE
                )

            truncateds = torch.full_like(terminals, t == self._rollout_length - 1)
            self.model_buffer.extend({
                'observations': states,
                'actions': actions,
                'next_observations': next_states,
                'rewards': rewards,
                'terminals': terminals,
                'truncateds': truncateds
            })
            steps_added += len(states)

            if terminals.all():
                break
            else:
                states = next_states[~terminals]

        mean_rollout_length = steps_added / self.cfg.rollout_batch_size
        self.log(f'Added {steps_added} transitions. Buffer size {len(self.model_buffer)}/{self.model_buffer.capacity}')
        self.log(f'Mean rollout length = {mean_rollout_length}')

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
        n_updates = counters['updates']
        epoch = n_updates // cfg.epoch_length

        if n_updates % cfg.epoch_length == 0:
            self.log(f'Starting epoch {epoch}')
            # Resize buffer to fit samples
            self._set_rollout_length(epoch)
            self._reallocate_model_buffer()

        if n_updates % cfg.model_train_period == 0:
            self._train_models(data)

            # Generate rollouts
            initial_states = data.sample(cfg.rollout_batch_size, replace=True)['observations']
            self._mb_rollouts(initial_states)

        # Update agent
        for _ in range(cfg.solver_updates):
            batch = self._get_mixed_batch(data)
            self.solver.update_with_batch(batch, counters)

    def additional_diagnostics(self, data: ReplayBuffer) -> dict:
        diagnostics = super().additional_diagnostics(data)
        solver_diagnostics = self.solver.additional_diagnostics(data)
        diagnostics.update(prefix_dict_keys('solver', solver_diagnostics))
        diagnostics['rollout_length'] = self._rollout_length

        def compute_model_metrics(*args):
            s, a, ns, r, t = [arg.unsqueeze(1) for arg in args]
            t = t.float()
            model_indices = [random.choice(self._elite_indices)]
            ns_distr, r_distr = self.model_ensemble.distribution(s, a, model_indices)
            t_distr = self.model_ensemble.terminal_distribution(ns, model_indices)
            return {
                'next_state_log_prob': ns_distr.log_prob(ns),
                'reward_log_prob': r_distr.log_prob(r),
                'terminal_log_prob': t_distr.log_prob(t),
                'next_state_error': torch.abs(ns_distr.loc - ns).mean(-1),
                'reward_error': torch.abs(r_distr.loc - r),
                'terminal_error': torch.abs(t_distr.probs - t)
            }

        with torch.no_grad():
            model_diagnostics = batch_map(
                compute_model_metrics,
                data.get('observations', 'actions', 'next_observations', 'rewards', 'terminals'),
                batch_size=BATCH_MAP_SIZE
            )
            model_diagnostics = {k: v.mean() for k, v in model_diagnostics.items()}
            model_diagnostics['log_likelihood'] = self._model_log_likelihood
            model_diagnostics['elite_log_likelihood'] = self._elite_log_likelihood
            model_diagnostics['min_logstd'] = self.model_ensemble.min_logstd.mean()
            model_diagnostics['max_logstd'] = self.model_ensemble.max_logstd.mean()
        diagnostics.update(prefix_dict_keys('model', model_diagnostics))

        return diagnostics
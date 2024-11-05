import itertools
import random
import time

import torch

from force.alg.agent import BufferedAgent
from force.alg.sac import SAC
from force.config import Field
from force.data import TransitionBuffer
from force.nn.layers import Linear
from force.nn.models.gaussian_dynamics_ensemble import GaussianDynamicsEnsemble
from force.nn.util import get_device, freepeat, batch_map
from force.util import prefix_dict_keys, pymean


MODEL_TRAIN_PRINT_PERIOD = 30.0 # in seconds

MUJOCO_TARGET_ENTROPIES = {
    'Hopper': -1.0, 'HalfCheetah': -3.0, 'Walker2d': -3.0, 'Ant': -4.0, 'Humanoid': -2.0
}


class MBPO(BufferedAgent):
    class Config(BufferedAgent.Config):
        solver = SAC.Config
        solver_updates = 20
        model_ensemble = GaussianDynamicsEnsemble.Config
        num_elites = 5
        epoch_length = 1000
        model_train_period = 250
        max_model_epochs = Field(int, required=False)
        max_model_epochs_since_update = 5
        rollout_schedule = [20, 100, 1, 1]
        rollout_batch_size = 100000
        model_retain_epochs = 1
        real_frac = 0.1

    def __init__(self, cfg, obs_space, act_space,
                 solver=None,
                 device=None,
                 termination_fn=None):
        assert cfg.num_elites <= cfg.model_ensemble.num_models
        assert 0 < cfg.real_frac < 1
        super().__init__(cfg, obs_space, act_space, device=device)

        if solver is None:
            solver = SAC(cfg.solver, obs_space, act_space, device=self.device)
        self.solver = solver
        self.model_ensemble = GaussianDynamicsEnsemble(
            cfg.model_ensemble, obs_space, act_space,
            device=self.device, termination_fn=termination_fn
        )

        # Variables for model training
        num_models = self.model_ensemble.num_models
        self._linear_layers = [
            [m for m in self.model_ensemble.models[i].modules() if isinstance(m, Linear)]
            for i in range(num_models)
        ]
        self._best_parameters = [
            None for _ in range(num_models)
        ]
        self._elite_indices = None
        self._best_lls = torch.full([num_models], float('-inf'))

    def act(self, obs, eval: bool):
        return self.solver.act(obs, eval)

    def _save_parameters(self, index):
        params = []
        for m in self._linear_layers[index]:
            params.append((
                m.weight.clone().detach(),
                m.bias.clone().detach()
            ))
        self._best_parameters[index] = params

    def _load_best_parameters(self):
        for i in range(self.model_ensemble.num_models):
            for m, params in zip(self._linear_layers[i], self._best_parameters[i]):
                m.weight.data, m.bias.data = params

    def _train_models(self):
        self.log('Training models...')
        dataset = self.replay_buffer.get(
            'observations', 'actions', 'next_observations', 'rewards',
            as_dict=True
        )

        # Re-fit model's normalizer to all states
        self.model_ensemble.normalizer.fit(dataset['observations'])

        # Repeat
        num_models = self.model_ensemble.num_models
        dataset_repeated = {k: freepeat(v, num_models, dim=1) for k, v in dataset.items()}

        def compute_ll():
            with torch.no_grad():
                return batch_map(
                    self.model_ensemble.log_likelihood,
                    dataset_repeated,
                    batch_device=self.device
                ).mean(0).cpu()

        last_print_t = time.time()
        epochs_since_update = 0
        for epoch in itertools.count():
            if self.cfg.max_model_epochs is not None and epoch >= self.cfg.max_model_epochs:
                break

            self.model_ensemble.epoch(dataset)
            lls = compute_ll()

            updated = False
            for i in range(num_models):
                if lls[i] > self._best_lls[i]:
                    self._best_lls[i] = lls[i]
                    updated = True
                    self._save_parameters(i)

            # print if enough time has passed since last print
            current_t = time.time()
            if current_t - last_print_t > MODEL_TRAIN_PRINT_PERIOD:
                self.log(f'Model epoch {epoch}: {lls.mean().item():.2f} (best: {self._best_lls.mean().item():.2f})')
                last_print_t = current_t

            if updated:
                epochs_since_update = 0
            else:
                epochs_since_update += 1

            if epochs_since_update > self.cfg.max_model_epochs_since_update:
                break

        self._load_best_parameters()

        # Check we reproduce best lls
        # final_lls = compute_ll()
        self.log(f'Log-likelihoods after training: {self._best_lls}')

        elite_info = torch.topk(self._best_lls, self.cfg.num_elites)
        self._elite_indices = elite_info.indices.tolist()
        self._model_log_likelihood = self._best_lls.mean()
        self._elite_log_likelihood = elite_info.values.mean()

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
        mb_buffer = self.solver.replay_buffer

        rollouts_per_epoch = cfg.epoch_length / cfg.model_train_period
        steps_per_rollout = cfg.rollout_batch_size * self._rollout_length
        steps_per_epoch = int(rollouts_per_epoch * steps_per_rollout)
        new_capacity = cfg.model_retain_epochs * steps_per_epoch
        if mb_buffer.capacity != new_capacity:
            self.log(f'Updating model buffer: {mb_buffer.capacity} -> {new_capacity}')
            samples = mb_buffer.get(as_dict=True)
            new_buffer = TransitionBuffer(self.obs_space, self.act_space, new_capacity,
                                          device=mb_buffer.device)
            new_buffer.extend(**samples)
            assert len(mb_buffer) == len(new_buffer)
            self.solver.replay_buffer = new_buffer

    def _mb_transition(self, states):
        actions = self.solver.act(states, eval=False)
        next_states, rewards, terminals = self.model_ensemble.sample(
            states, actions, model_index=random.choice(self._elite_indices)
        )
        return actions, next_states, rewards, terminals

    def _mb_rollouts(self):
        mb_buffer = self.solver.replay_buffer
        states = self.replay_buffer.sample(self.cfg.rollout_batch_size, replace=True)['observations']
        steps_added = 0
        for t in range(self._rollout_length):
            with torch.no_grad():
                actions, next_states, rewards, terminals = batch_map(
                    self._mb_transition, states
                )

            truncateds = torch.full_like(terminals, t == self._rollout_length - 1)
            mb_buffer.extend(
                observations=states,
                actions=actions,
                next_observations=next_states,
                rewards=rewards,
                terminals=terminals,
                truncateds=truncateds
            )
            steps_added += len(states)

            if terminals.all():
                break
            else:
                states = next_states[~terminals]

        mean_rollout_length = steps_added / self.cfg.rollout_batch_size
        self.log(f'Added {steps_added} transitions. Buffer size {len(mb_buffer)}/{mb_buffer.capacity}')
        self.log(f'Mean rollout length = {mean_rollout_length}')

    def _get_mixed_batch(self):
        batch_size = self.solver.cfg.batch_size
        num_real = int(self.cfg.real_frac * batch_size)
        real_batch = self.replay_buffer.sample(num_real)
        model_batch = self.solver.replay_buffer.sample(batch_size - num_real)
        return {
            k: torch.cat([real_batch[k], model_batch[k]]).to(self.device)
            for k in real_batch.keys()
        }

    def _update(self, counters):
        cfg = self.cfg
        n_updates = counters['updates']
        epoch = n_updates // cfg.epoch_length

        if n_updates % cfg.epoch_length == 0:
            self.log(f'Starting epoch {epoch}')
            # Resize buffer to fit samples
            self._set_rollout_length(epoch)
            self._reallocate_model_buffer()

        if n_updates % cfg.model_train_period == 0:
            self._train_models()
            self._mb_rollouts()

        # Update agent
        for _ in range(cfg.solver_updates):
            batch = self._get_mixed_batch()
            self.solver.update_with_minibatch(batch, counters)

    def additional_diagnostics(self) -> dict:
        diagnostics = super().additional_diagnostics()

        solver_diagnostics = self.solver.additional_diagnostics()
        diagnostics.update(prefix_dict_keys('solver', solver_diagnostics))
        diagnostics['rollout_length'] = self._rollout_length

        def compute_model_metrics(*args):
            s, a, ns, r, t = args
            model_indices = [random.choice(self._elite_indices)]
            ns_distr, r_distr = self.model_ensemble.distribution(s.unsqueeze(1), a.unsqueeze(1), model_indices)
            ns_mode = ns_distr.mode.squeeze(1)
            r_mode = r_distr.mode.squeeze(1)
            t2 = self.model_ensemble.termination_fn(ns)
            ns_hat_t = self.model_ensemble.termination_fn(ns_mode)
            t_error = (t2 != t)

            return {
                'next_state_log_prob': ns_distr.log_prob(ns.unsqueeze(1)),
                'reward_log_prob': r_distr.log_prob(r.unsqueeze(1)),
                # 'terminal_log_prob': t_distr.log_prob(t),
                'next_state_error': torch.abs(ns_mode - ns).mean(-1),
                'reward_error': torch.abs(r_mode - r),
                'terminal_error': t_error.float(),
                'predicted_terminal_error': (ns_hat_t != t).float()
            }

        with torch.no_grad():
            model_diagnostics = batch_map(
                compute_model_metrics,
                self.replay_buffer.get('observations', 'actions', 'next_observations', 'rewards', 'terminals'),
                batch_device=self.device
            )
            model_diagnostics = {k: v.mean() for k, v in model_diagnostics.items()}
            model_diagnostics['log_likelihood'] = self._model_log_likelihood
            model_diagnostics['elite_log_likelihood'] = self._elite_log_likelihood
            model_diagnostics['min_logstd'] = self.model_ensemble.min_logstd.mean()
            model_diagnostics['max_logstd'] = self.model_ensemble.max_logstd.mean()
        diagnostics.update(prefix_dict_keys('model', model_diagnostics))

        return diagnostics
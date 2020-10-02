from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from force.log import default_logger as log
from force.normalization import Normalizer
from force.train import get_optimizer, epochal_training, supervised_loss, L2Loss
from force.torch_util import device, Module, mlp, random_indices
from force.etc.batch_ensemble import BatchEnsemble


class BaseModel(ABC):
    @abstractmethod
    def __call__(self, states, actions):
        pass


class DynamicsModel(Module, BaseModel):
    def __init__(self, state_dim, action_dim, hidden_dims, predict_reward=False, layer_class=nn.Linear):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.predict_reward = predict_reward

        self.state_normalizer = Normalizer(state_dim)
        self.diff_normalizer = Normalizer(state_dim)

        if predict_reward:
            self.reward_normalizer = Normalizer(1)
            output_dim = state_dim + 1
        else:
            output_dim = state_dim

        input_dim = state_dim + action_dim
        self.net = mlp([input_dim, *hidden_dims, output_dim], layer_class=layer_class)
        self.compute_loss = supervised_loss(self.net, L2Loss())
        self.optimizer = get_optimizer(self.net)

    def fit(self, buffer, epochs, **kwargs):
        states, actions, next_states, rewards, _ = buffer.get()
        diffs = next_states - states

        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)

        self.state_normalizer.fit(states)
        self.diff_normalizer.fit(diffs)
        normalized_states = self.state_normalizer(states)
        normalized_diffs = self.diff_normalizer(diffs)

        if self.predict_reward:
            self.reward_normalizer.fit(rewards)
            normalized_rewards = self.reward_normalizer(rewards)

        inputs = torch.cat([normalized_states, actions], dim=1)
        targets = torch.cat([normalized_diffs, normalized_rewards], dim=1) if self.predict_reward else normalized_diffs
        epochal_training(self.compute_loss, self.optimizer, [inputs, targets], epochs, **kwargs)

    def __call__(self, states, actions):
        normalized_states = self.state_normalizer(states)
        with torch.no_grad():
            outputs = self.net(torch.cat([normalized_states, actions], dim=1))
        if self.predict_reward:
            diffs = self.diff_normalizer.unnormalize(outputs[:,:self.state_dim])
            rewards = self.reward_normalizer.unnormalize(outputs[:,self.state_dim])
            return states + diffs, rewards
        else:
            diffs = self.diff_normalizer.unnormalize(outputs)
            return states + diffs


class ModelEnsemble(Module, BaseModel):
    def __init__(self, ensemble_size, state_dim, action_dim, hidden_dims,
                 predict_reward=False, layer_class=nn.Linear,
                 n_pick=None):
        super().__init__()
        self.models = nn.ModuleList([
            DynamicsModel(state_dim, action_dim, hidden_dims=hidden_dims,
                         predict_reward=predict_reward, layer_class=layer_class) \
            for _ in range(ensemble_size)
        ])
        self.ensemble_size = ensemble_size
        self.n_pick = n_pick if n_pick is not None else ensemble_size

    def pick_indices(self):
        return random_indices(self.ensemble_size, self.n_pick, replace=False)

    def pick_models(self):
        return [self.models[i] for i in self.pick_indices()]

    def fit(self, dataset, epochs, verbose=False, **kwargs):
        for i, model in enumerate(self.models):
            if verbose:
                log.message(f'Fitting model {i+1}/{self.ensemble_size}')
            model.fit(dataset, epochs, verbose=verbose, **kwargs)

    def __call__(self, states, actions):
        outputs = [model(states, actions) for model in self.pick_models()]
        if len(outputs) == 1:
            return outputs[0]
        elif isinstance(outputs[0], tuple): # joint model
            n_outputs = len(outputs[0])
            return tuple(torch.stack([output[i] for output in outputs], dim=0).mean(dim=0) for i in range(n_outputs))
        else:
            return np.mean(outputs, axis=0)


class BatchEnsembleDynamicsModel(Module, BaseModel):
    def __init__(self, ensemble_size, state_dim, action_dim, hidden_dims, predict_reward=False):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.predict_reward = predict_reward

        self.state_normalizer = Normalizer(state_dim)
        self.diff_normalizer = Normalizer(state_dim)

        if predict_reward:
            self.reward_normalizer = Normalizer(1)
            output_dim = state_dim + 1
        else:
            output_dim = state_dim

        input_dim = state_dim + action_dim
        self.net = BatchEnsemble(ensemble_size, [input_dim, *hidden_dims, output_dim]).to(device)

        self.compute_loss = self.net.loss_closure(L2Loss())
        self.optimizer = get_optimizer(self.net.param_groups())

    def fit(self, buffer, epochs, **kwargs):
        states, actions, next_states, rewards, _ = buffer.get()
        diffs = next_states - states

        if rewards.ndim == 1:
            rewards = np.expand_dims(rewards, 1)

        self.state_normalizer.fit(states)
        self.diff_normalizer.fit(diffs)
        normalized_states = self.state_normalizer(states)
        normalized_diffs = self.diff_normalizer(diffs)
        if self.predict_reward:
            self.reward_normalizer.fit(rewards)
            normalized_rewards = self.reward_normalizer(rewards)

        inputs = np.hstack([normalized_states, actions])
        targets = np.hstack([normalized_diffs, normalized_rewards]) if self.predict_reward else normalized_diffs
        epochal_training(self.compute_loss, self.optimizer, [inputs, targets], epochs, **kwargs)

    def __call__(self, states, actions):
        normalized_states = self.state_normalizer(states)
        with torch.no_grad():
             outputs = self.net(torch.cat([normalized_states, actions], dim=1), repeat=False, split=False).cpu().numpy()
        if self.predict_reward:
            diffs = self.diff_normalizer.unnormalize(outputs[:,:self.state_dim])
            rewards = self.reward_normalizer.unnormalize(outputs[:,self.state_dim])
            return states + diffs, rewards
        else:
            diffs = self.diff_normalizer.unnormalize(outputs)
            return states + diffs


class OracleDynamics(BaseModel):
    def __init__(self, env_class):
        assert hasattr(env_class, 'oracle_dynamics')
        self.env = env_class()

    def __call__(self, states, actions):
        if states.ndim == 1 and actions.ndim == 1:
            return self.env.oracle_dynamics(states, actions)
        else:
            next_states, rewards = [], []
            for s, a in zip(states, actions):
                next_state, reward = self.env.oracle_dynamics(s, a)
                next_states.append(next_state)
                rewards.append(reward)
            return np.array(next_states), np.array(rewards)
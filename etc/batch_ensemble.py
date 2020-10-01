import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from force.torch_util import device, Module, torchify


import pdb

class BatchEnsembleLayer(nn.Module):
    def __init__(self, ensemble_size, in_features, out_features, bias=True):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.in_features = in_features
        self.out_features = out_features
        self.slow_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.R = nn.Parameter(torch.Tensor(ensemble_size, in_features))
        self.S = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.slow_weight, a=math.sqrt(5))
        for fast_weight in (self.R, self.S):
            # Initialize with random signs
            fast_weight.data.copy_(torch.sign(torch.normal(mean=torch.zeros_like(fast_weight), std=1)))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.slow_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        effective_batch_size = len(x)
        assert effective_batch_size % self.ensemble_size == 0
        batch_size = effective_batch_size // self.ensemble_size
        if batch_size == 1:
            R_rep, S_rep, bias_rep = self.R, self.S, self.bias
        else:
            R_rep = self.R.repeat_interleave(batch_size, dim=0)
            S_rep = self.S.repeat_interleave(batch_size, dim=0)
            bias_rep = self.bias.repeat_interleave(batch_size, dim=0)
        return F.linear(x * R_rep, self.slow_weight, bias_rep) * S_rep

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class BatchEnsemble(Module):
    def __init__(self, ensemble_size, dims, bias=True, activation=nn.ReLU(), output_activation=None):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.layers = nn.ModuleList([
            BatchEnsembleLayer(ensemble_size, dims[i-1], dims[i], bias=bias) \
            for i in range(1, len(dims))
        ])
        self.activation = activation
        self.output_activation = output_activation

    def slow_weights(self):
        for layer in self.layers:
            yield layer.slow_weight

    def fast_weights(self):
        for layer in self.layers:
            yield layer.R
            yield layer.S

    def biases(self):
        for layer in self.layers:
            yield layer.bias

    def param_groups(self):
        return (
            {'params': self.slow_weights()},
            {'params': self.fast_weights(), 'weight_decay': 0},
            {'params': self.biases(), 'weight_decay': 0},
        )

    def forward(self, x, repeat=False, split=True):
        if repeat:
            inputs_per_member = len(x)
            x = x.repeat(self.ensemble_size, 1)
        else:
            inputs_per_member = len(x) // self.ensemble_size

        for layer in self.layers:
            x = self.activation(layer(x))
        if self.output_activation is not None:
            x = self.output_activation(x)

        return torch.split(x, inputs_per_member) if split else x

    def average(self, x):
        return torch.mean(torch.stack(self(x, repeat=True, split=True), dim=0), 0)

    def loss_closure(self, criterion):
        return lambda x, y: criterion(self(x, repeat=False, split=False), y)


if __name__ == '__main__':
    from force.policy import RandomPolicy
    from force.sampling import StepSampler
    from force.train import epochal_training, get_optimizer, L2Loss
    from force.util import batch_map
    from force.env.util import get_env, env_dims
    env = get_env('halfcheetah')
    state_dim, action_dim = env_dims(env)

    def get_data(n):
        data = StepSampler(env).run(RandomPolicy(env.action_space), n)
        states, actions, next_states, rewards, dones = data.get()
        sa = torch.cat([states, actions], dim=1)
        diffs = next_states - states
        return sa, diffs

    batch_ensemble = BatchEnsemble(100, [state_dim+action_dim, 500, 500, state_dim]).to(device)
    train_sa, train_diffs = get_data(100000)
    test_sa, test_diffs = get_data(10000)

    def evaluate():
        print('Evaluating')
        for name, sa, diffs in (('train', train_sa, train_diffs), ('test', test_sa, test_diffs)):
            predictions = batch_map(lambda batch: batch_ensemble.average(torchify(batch)).detach(),
                                    sa,
                                    batch_size=100)
            errors = torch.norm(predictions - diffs, dim=1)
            print(f'Average L2 error {name}:', torch.mean(errors).cpu())

    evaluate()
    for _ in range(100):
        print('Training')
        epochal_training(batch_ensemble.loss_closure(L2Loss()),
                         optimizer=get_optimizer(batch_ensemble),
                         data=[train_sa, train_diffs],
                         epochs=1, batch_size=1000,
                         verbose=False, progress_bar=False)
        evaluate()
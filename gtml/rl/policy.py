import torch; nn = torch.nn


class Policy:
    def act(self, observations):
        raise NotImplementedError

    def act1(self, observation):
        return self.act(torch.unsqueeze(observation, 0))[0]


class ParametricPolicy(Policy):
    def __init__(self, net):
        self.net = net

    def parameters(self):
        return self.net.parameters()


class DeterministicPolicy(ParametricPolicy):
    def act(self, observations):
        return self.net(observations)


class StochasticPolicy(ParametricPolicy):
    def action_distributions(self, observations):
        raise NotImplementedError

    def act(self, observations):
        return self.action_distributions(observations).sample()


class CategoricalPolicy(StochasticPolicy):
    def action_distributions(self, observations):
        logits = self.net(observations)
        return torch.distributions.Categorical(logits=logits)


class GaussianPolicy(StochasticPolicy):
    def __init__(self, mean, var):
        ParametricPolicy.__init__(self, net)
        self.mean = mean
        self.var = var

    def action_distributions(self, observations):
        mean = self.mean(observations)
        var = self.var(observations)
        return torch.distributions.MultivariateNormal(mean, covariance_matrix=torch.diag(var))


class ActorCritic(nn.Module):
    def __init__(self, policy, value_fn):
        nn.Module.__init__(self)
        self.policy = policy
        self.policy_fn = policy.net
        self.value_fn = value_fn

    def forward(self, x):
        return self.policy_fn(x), self.value_fn(x)

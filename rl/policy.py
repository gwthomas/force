import torch


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
    def __init__(self, net, distribution):
        ParametricPolicy.__init__(self, net)
        self.distribution = distribution

    def action_distributions(self, observations):
        return self.distribution(self.net(observations))

    def act(self, observations):
        return self.action_distributions(observations).sample()


class CategoricalPolicy(StochasticPolicy):
    def __init__(self, net):
        def distribution(input):
            return torch.distributions.Categorical(logits=input)
        StochasticPolicy.__init__(self, net, distribution)


class GaussianPolicy(StochasticPolicy):
    def __init__(self, net):
        StochasticPolicy.__init__(self, net, torch.distributions.Normal)


class ActorCritic:
    def __init__(self, policy, value_fn):
        self.policy = policy
        self.value_fn = value_fn

    @property
    def actor(self):
        return self.policy

    @property
    def critic(self):
        return self.value_fn

    def parameters(self):
        # return self.policy.parameters() + self.value_fn.parameters()
        for param in self.policy.parameters():
            yield param
        for param in self.value_fn.parameters():
            yield param

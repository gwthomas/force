import numpy as np

from gtml.defaults import GAE_LAMBDA


# thx OpenAI
# computes discounted sums along 0th dimension of x
# returns an ndarray with same shape as x, satisfying
#    y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
# where k = len(x) - t - 1
def discount(x, gamma):
    assert x.ndim >= 1
    return lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

# thx OpenAI (adapted)
# computes target value using TD(lambda) estimator and advantage with GAE(lambda)
def estimate_advantages_and_value_targets(observations, actions, rewards, done, values,
        gamma, lam=GAE_LAMBDA):
    T = len(rewards)
    advantages = np.empty(T, 'float32')
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = int(not (done and t == T - 1))
        delta = rewards[t] + gamma * values[t+1] * nonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    return advantages, advantages + values[:-1]

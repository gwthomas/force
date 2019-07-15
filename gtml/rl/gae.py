import torch

from gtml.constants import DEFAULT_GAE_LAMBDA


# thx OpenAI (adapted)
# computes target value using TD(lambda) estimator and advantage with GAE(lambda)
def estimate_advantages_and_value_targets(rewards, done, values, gamma, lam=DEFAULT_GAE_LAMBDA):
    T = len(rewards)
    advantages = []
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = int(not (done and t == T - 1))
        delta = rewards[t] + gamma * values[t+1] * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        advantages.append(lastgaelam)
    advantages = torch.Tensor(advantages)
    return advantages, advantages + values[:-1]

import torch

from gtml.constants import DEFAULT_BATCH_SIZE


def test(model, test_set, batch_size=DEFAULT_BATCH_SIZE):
    model.eval()
    results = []
    for x, y in test_set:
        prediction = model(torch.unsqueeze(x,0)).argmax()
        results.append((prediction == y).item())
    model.train()
    return torch.Tensor(results).mean().item()

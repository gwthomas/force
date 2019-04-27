import torch
from torch.utils.data import DataLoader

from gtml.constants import DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS, DEVICE


def zero_one(y_hat, y):
    return y_hat == y

def test(model, dataset, criterion=zero_one, batch_size=DEFAULT_BATCH_SIZE,
         num_workers=DEFAULT_NUM_WORKERS):
    data_loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    with torch.no_grad():
        batch_results = []
        for batch in data_loader:
            batch = [item.to(DEVICE) for item in batch]
            x, y = batch
            y_hat = model(x).argmax(dim=1)
            batch_results.append(criterion(y_hat, y))
        all_results = torch.cat(batch_results)
        retval = all_results.float().mean().item()
    model.train()
    return retval

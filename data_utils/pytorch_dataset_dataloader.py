
from typing import *
import numpy as np
from data_utils.dataloader import Dataloader

if TYPE_CHECKING:
    import torch
else:
    try: import torch
    except: pass

# TODO: shuffling!

class PytorchDatasetDataloader(Dataloader):
    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int):
        self._dataset = dataset
        self._batch_size = batch_size
    def __iter__(self):
        for i in range(len(self)):
            start = i * self._batch_size
            end = min(start + self._batch_size, len(self._dataset))
            datapoints = [self._dataset[j] for j in range(start, end)]
            X_torch = torch.stack([X_pt for X_pt, Y_pt in datapoints])
            Y_torch = torch.stack([
                torch.tensor(Y_pt) if not isinstance(Y_pt, torch.Tensor) else Y_pt
                for X_pt, Y_pt in datapoints
            ])
            X_numpy, Y_numpy = X_torch.numpy(), Y_torch.numpy()
            yield X_numpy, Y_numpy
    def __len__(self):
        return (len(self._dataset) + self._batch_size - 1) // self._batch_size

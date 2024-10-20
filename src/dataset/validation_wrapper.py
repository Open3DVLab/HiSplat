from typing import Iterator, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset


class ValidationWrapper(Dataset):
    """Wraps a dataset so that PyTorch Lightning's validation step can be turned into a
    visualization step.
    """

    dataset: Dataset
    dataset_iterator: Optional[Iterator]
    length: int

    def __init__(self, dataset: Dataset, length: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.length = length
        self.dataset_iterator = None
        self.iter_num = 0

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        if isinstance(self.dataset, IterableDataset):
            # TODO: Very dangerous, may cause leaking
            try:
                world_size = dist.get_world_size()
            except:
                world_size = 1
            if self.dataset_iterator is None or self.iter_num >= self.length / world_size:
                self.iter_num = 0
                self.dataset_iterator = iter(self.dataset)
            self.iter_num += 1
            return next(self.dataset_iterator)

        random_index = torch.randint(0, len(self.dataset), tuple())
        return self.dataset[random_index.item()]

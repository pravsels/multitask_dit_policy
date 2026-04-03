from __future__ import annotations

import math

import torch


class DistributedIndexSampler(torch.utils.data.Sampler[int]):
    """Shard a precomputed list of indices across distributed ranks."""

    def __init__(
        self,
        indices: list[int],
        *,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.indices = indices
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.num_replicas = num_replicas or 1
        self.rank = rank or 0
        self.epoch = 0

        if self.drop_last and len(self.indices) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.indices) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.indices) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(self.indices), generator=generator).tolist()
            indices = [self.indices[i] for i in perm]
        else:
            indices = self.indices.copy()

        if self.drop_last:
            indices = indices[: self.total_size]
        else:
            padding = self.total_size - len(indices)
            if padding > 0:
                indices += indices[:padding]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

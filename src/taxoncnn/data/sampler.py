import torch
import math
import numpy as np
from collections import defaultdict

class ShardBatchSampler(torch.utils.data.Sampler):
    """
    BatchSampler that groups samples by shard for efficient cache reuse

    Ensures each batch contains samples from a single shard, reducing RAM usage and improving cache efficiency

    Args:
        dataset (ShardedCFTorchDataset): Dataset with shard indexing
        batch_size (int): Number of samples per batch
        shuffle (bool, optional): Whether to shuffle shards and samples. Defaults to True
        drop_last (bool, optional): Whether to drop the last incomplete batch in each shard. Defaults to False
        seed (int or None, optional): Random seed for shuffling. Defaults to None
    """
    def __init__(self, dataset, batch_size,
                 shuffle=True, drop_last=False, seed=None):
        self.ds = dataset
        self.bs = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.rng = np.random.default_rng(seed)
        # build per-shard index lists
        per = defaultdict(list)
        for gi, (si, _local) in enumerate(self.ds.index):
            per[si].append(gi)
        self.per_shard = dict(per)

    def __iter__(self):
        """
        Yield batches of indices, each batch containing samples from a single shard

        Yields:
            list[int]: List of sample indices for a batch
        """
        shard_ids = list(self.per_shard.keys())
        if self.shuffle:
            self.rng.shuffle(shard_ids)
        for si in shard_ids:
            idxs = self.per_shard[si][:]
            if self.shuffle:
                self.rng.shuffle(idxs)
            n = len(idxs)
            limit = (n // self.bs) * self.bs if self.drop_last else n
            for k in range(0, limit, self.bs):
                yield idxs[k:k+self.bs]

    def __len__(self):
        """
        Returns the total number of batches

        Returns:
            int: Number of batches
        """
        n = 0
        for idxs in self.per_shard.values():
            n += (len(idxs) // self.bs) if self.drop_last else math.ceil(len(idxs) / self.bs)
        return n
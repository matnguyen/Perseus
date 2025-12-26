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
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, seed=None):
        self.ds = dataset
        self.bs = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.rng = np.random.default_rng(seed)

        # number of shards
        if hasattr(self.ds, "sizes"):
            self.n_shards = len(self.ds.sizes)
        elif hasattr(self.ds, "paths"):
            self.n_shards = len(self.ds.paths)
        else:
            raise ValueError("Dataset must expose .sizes or .paths to infer #shards")

    def __iter__(self):
        shard_ids = np.arange(self.n_shards, dtype=np.int32)
        if self.shuffle:
            self.rng.shuffle(shard_ids)

        for si in shard_ids.tolist():
            # local indices allowed for this split
            loc = self.ds.allowed_local_indices(si)

            # loc can be numpy array or torch tensor; normalize to numpy int64
            if isinstance(loc, torch.Tensor):
                loc = loc.cpu().numpy()
            loc = np.asarray(loc)

            if loc.size == 0:
                continue

            if self.shuffle:
                self.rng.shuffle(loc)

            n = int(loc.size)
            limit = (n // self.bs) * self.bs if self.drop_last else n

            base = int(self.ds.offsets[si])  # global offset for shard si

            for k in range(0, limit, self.bs):
                batch_locals = loc[k:k + self.bs]
                # map to global dataset indices
                yield (base + batch_locals).tolist()

    def __len__(self):
        # Conservative length computation: sum batches per shard based on allowed count
        total = 0
        for si in range(self.n_shards):
            loc = self.ds.allowed_local_indices(si)
            if isinstance(loc, torch.Tensor):
                n = int(loc.numel())
            else:
                n = int(np.asarray(loc).size)

            if n == 0:
                continue
            total += (n // self.bs) if self.drop_last else math.ceil(n / self.bs)
        return total
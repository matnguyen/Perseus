import torch
import math
import numpy as np
from collections import defaultdict

import torch
import math
import numpy as np

class ShardBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler that (a) groups batches within a shard for cache reuse and
    (b) optionally does epoch-wise resampling:
        - choose only K shards per epoch
        - choose up to M samples per shard per epoch

    Call `sampler.set_epoch(epoch)` each epoch (your Trainer likely already does this).
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        drop_last=False,
        seed=0,
        shards_per_epoch=None,      # NEW: int or None
        samples_per_shard=None,     # NEW: int or None
        val_shards_per_epoch=None,  # NEW: int or None
        val_samples_per_shard=None, # NEW: int or None
        replace_shards=False,       # NEW: allow sampling shards with replacement (usually False)
        replace_samples=False,      # NEW: allow sampling samples with replacement (usually False)
    ):
        self.ds = dataset
        self.bs = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        self.shards_per_epoch = None if shards_per_epoch is None else int(shards_per_epoch)
        self.samples_per_shard = None if samples_per_shard is None else int(samples_per_shard)
        self.val_shards_per_epoch = None if val_shards_per_epoch is None else int(val_shards_per_epoch)
        self.val_samples_per_shard = None if val_samples_per_shard is None else int(val_samples_per_shard)
        self.replace_shards = bool(replace_shards)
        self.replace_samples = bool(replace_samples)
        
        if val_shards_per_epoch is not None or val_samples_per_shard is not None:
            self.is_val = True
        else:
            self.is_val = False

        if hasattr(self.ds, "sizes"):
            self.n_shards = len(self.ds.sizes)
        elif hasattr(self.ds, "paths"):
            self.n_shards = len(self.ds.paths)
        else:
            raise ValueError("Dataset must expose .sizes or .paths to infer #shards")

        # RNG is re-seeded per epoch in set_epoch()
        self.rng = np.random.default_rng(self.seed)

        # Cache: allowed counts per shard (all allowed indices, not per-epoch subsample)
        self._allowed_counts = None

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        # epoch-dependent RNG so each epoch uses a different shard/sample subset deterministically
        self.rng = np.random.default_rng(self.seed + self.epoch)

    def _epoch_shard_ids(self) -> np.ndarray:
        shard_ids = np.arange(self.n_shards, dtype=np.int32)

        k = self.val_shards_per_epoch if self.is_val else self.shards_per_epoch
        if k is not None and k < self.n_shards:
            shard_ids = self.rng.choice(
                shard_ids,
                size=k,
                replace=self.replace_shards,
            ).astype(np.int32, copy=False)

        if self.shuffle:
            shard_ids = shard_ids.copy()
            self.rng.shuffle(shard_ids)

        return shard_ids

    def __iter__(self):
        shard_ids = self._epoch_shard_ids()

        for si in shard_ids.tolist():
            loc = self.ds.allowed_local_indices(si)
            if isinstance(loc, torch.Tensor):
                loc = loc.cpu().numpy()
            loc = np.asarray(loc, dtype=np.int64)

            if loc.size == 0:
                continue

            # epoch-wise sample subsample within shard
            m = self.val_samples_per_shard if self.is_val else self.samples_per_shard
            if m is not None and m < loc.size:
                loc = self.rng.choice(
                    loc,
                    size=m,
                    replace=self.replace_samples,
                ).astype(np.int64, copy=False)

            if self.shuffle:
                loc = loc.copy()  # avoid permuting dataset cache
                self.rng.shuffle(loc)

            n = int(loc.size)
            if n == 0:
                continue

            limit = (n // self.bs) * self.bs if self.drop_last else n
            base = int(self.ds.offsets[si])

            for k in range(0, limit, self.bs):
                batch_locals = loc[k:k + self.bs]
                if self.drop_last and batch_locals.size < self.bs:
                    continue
                yield (base + batch_locals).tolist()

    def __len__(self):
        """
        Important: With epoch-wise resampling, the true number of batches varies by epoch
        (and depends on how many allowed indices each chosen shard has).

        DataLoader does not strictly require an exact length, but trainers/progress bars like it.
        So:
          - if no resampling: return exact
          - if resampling: return a stable estimate
        """
        # No epoch-wise resampling => exact length (cached)
        if self.shards_per_epoch is None and self.samples_per_shard is None:
            if self._allowed_counts is None:
                counts = np.zeros(self.n_shards, dtype=np.int64)
                for si in range(self.n_shards):
                    loc = self.ds.allowed_local_indices(si)
                    if isinstance(loc, torch.Tensor):
                        counts[si] = int(loc.numel())
                    else:
                        counts[si] = int(np.asarray(loc).size)
                self._allowed_counts = counts

            total_batches = 0
            for n in self._allowed_counts.tolist():
                if n == 0:
                    continue
                total_batches += (n // self.bs) if self.drop_last else math.ceil(n / self.bs)
            return int(total_batches)

        # With resampling => estimate length
        # Estimate allowed per shard using cached counts (build once)
        if self._allowed_counts is None:
            counts = np.zeros(self.n_shards, dtype=np.int64)
            for si in range(self.n_shards):
                loc = self.ds.allowed_local_indices(si)
                if isinstance(loc, torch.Tensor):
                    counts[si] = int(loc.numel())
                else:
                    counts[si] = int(np.asarray(loc).size)
            self._allowed_counts = counts

        # expected shards per epoch
        k = self.shards_per_epoch if self.shards_per_epoch is not None else self.n_shards

        # expected samples per shard per epoch
        if self.samples_per_shard is None:
            # use mean allowed count as a rough estimator
            mean_allowed = float(self._allowed_counts.mean()) if self._allowed_counts.size else 0.0
            per_shard = mean_allowed
        else:
            per_shard = float(self.samples_per_shard)

        # batches per shard
        if self.drop_last:
            batches_per_shard = int(per_shard // self.bs)
        else:
            batches_per_shard = int(math.ceil(per_shard / self.bs)) if per_shard > 0 else 0

        return int(k * batches_per_shard)
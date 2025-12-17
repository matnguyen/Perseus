import os
import time
import gc
import json
import logging
from pathlib import Path
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader

from perseus.data.collate import PadMaskCollateCF
from perseus.data.sampler import ShardBatchSampler
from perseus.utils.constants import (
    CANONICAL_RANKS,
    CROP_MAX_T
)
from perseus.trainer.utils import (
    normalize_y_per_rank_to7,
    remap_rank_index_to7,
    build_rank_filtered_index
)

logger = logging.getLogger(__name__)

class ShardedCFTorchDataset(Dataset):
    """
    PyTorch Dataset for channel-first .pt shards with Option-B labels

    Loads shards containing:
        - x: [N, C, T] (features)
        - y: [N] (legacy, mirror of y_any)
        - y_any: [N] (float)
        - y_rank: [N] (float)
        - y_per_rank: [N, R] (float, optional)
        - rank_index: [N] (int)
        - seq_id: [N] (optional)
        - taxon: [N] (optional)
        - lengths: [N] (optional)

    Supports per-worker LRU shard cache, optional fp16 downcast, and aggressive eviction

    Args:
        manifest_or_dir (str or Path): Directory of .pt shards or a manifest .json file
        subset_index (list, optional): List of (shard_idx, local_idx) tuples for subset selection
        cache_shards (int, optional): Number of shards to cache in memory per worker
        to_float32 (bool, optional): Convert features to float32 on load
        downcast_cache_dtype (str or None, optional): Downcast cache dtype ("float16" or None)
    """
    def __init__(self, manifest_or_dir, subset_index=None, cache_shards=1,
                 to_float32=False, downcast_cache_dtype="float16"):
        p = Path(manifest_or_dir)
        if p.is_dir():
            self.paths = sorted(str(x) for x in p.glob("*.pt"))
        elif p.suffix.lower() == ".json":
            mani = json.loads(p.read_text())
            # Prefer manifests that include sizes to avoid loading every shard
            self.paths = mani.get("outputs", [])
            self._sizes = mani.get("sizes", None)  # optional: list of N per shard
        else:
            raise ValueError("Pass a shard directory or permute_manifest.json")
        if not self.paths:
            raise FileNotFoundError(f"No shard .pt files under {manifest_or_dir}")

        self._cache = OrderedDict()
        self._cache_cap = max(1, int(cache_shards))
        self.to_float32 = bool(to_float32)
        self.downcast_cache_dtype = downcast_cache_dtype  # "float16" or None
        self.base_dir = os.path.dirname(manifest_or_dir)

        # Build index
        if subset_index is not None:
            self.index = list(subset_index)
        else:
            self.index = []
            if getattr(self, "_sizes", None):
                # Use sizes from manifest without opening shards
                for si, n in enumerate(self._sizes):
                    self.index.extend((si, j) for j in range(int(n)))
            else:
                # Fallback: read N from each shard (slower; small memory spikes)
                for si, path in enumerate(self.paths):
                    m = torch.load(os.path.join(self.base_dir, os.path.basename(path)), map_location="cpu")
                    n = int(m["x"].shape[0])
                    self.index.extend((si, j) for j in range(n))
        logger.info("Dataset: %d shards, %d samples (cache_shards=%d, downcast=%s)",
                 len(self.paths), len(self.index), self._cache_cap, self.downcast_cache_dtype or "none")

    def __len__(self): 
        """
        Returns the number of samples in the dataset

        Returns:
            int: Number of samples
        """
        return len(self.index)

    def _downcast_inplace(self, m):
        """
        Downcast tensors in the shard dict to the specified dtype in-place

        Args:
            m (dict): Shard dictionary

        Returns:
            dict: Shard dictionary with downcasted tensors
        """
        def _maybe(t):
            if isinstance(t, torch.Tensor) and t.dtype == torch.float32 and self.downcast_cache_dtype == "float16":
                return t.half()
            return t
        if "x" in m and isinstance(m["x"], torch.Tensor):
            m["x"] = _maybe(m["x"])
        if "x_list" in m and isinstance(m["x_list"], (list, tuple)):
            m["x_list"] = [_maybe(t) for t in m["x_list"]]
        return m

    def _evict_one(self):
        """
        Evict the least recently used shard from the cache and free memory
        """
        evict_si, evict_m = self._cache.popitem(last=False)
        # Break references for large tensors so storage is released
        for k in ("x","x_list"):
            if k in evict_m:
                evict_m[k] = None
        del evict_m
        gc.collect()

    def _get_shard(self, si: int):
        """
        Load a shard by index, using the cache if possible

        Args:
            si (int): Shard index

        Returns:
            dict: Loaded shard dictionary
        """
        if si in self._cache:
            self._cache.move_to_end(si)
            return self._cache[si]
        t0 = time.perf_counter()
        m = torch.load(os.path.join(self.base_dir, os.path.basename(self.paths[si])), map_location="cpu")
        if self.downcast_cache_dtype:
            m = self._downcast_inplace(m)
        self._cache[si] = m
        if len(self._cache) > self._cache_cap:
            self._evict_one()
        logger.debug("Shard cache: loaded si=%d in %.3fs", si, time.perf_counter() - t0)
        return m
    
    def __getitem__(self, i):
        """
        Get a sample by global index

        Args:
            i (int): Sample index

        Returns:
            dict: Sample dictionary with keys:
                - "x": Tensor [C, T]
                - "y_any": float
                - "y_rank": float
                - "y_per_rank": Tensor [7]
                - "rank_index": int (0..6 or -1)
                - "length": int
                - "seq_id": str or None
                - "taxon": str or None
        """
        si, j = self.index[i]
        m = self._get_shard(si)

        x = (m["x"][j] if "x" in m else m["x_list"][j])  # [C,T]
        if self.to_float32 and x.dtype != torch.float32:
            x = x.float()

        y_any   = m.get("y_any", m.get("y"))[j].float()

        y_rankT = m.get("y_rank", torch.zeros((), dtype=torch.float32)).reshape(-1)
        y_rank  = (y_rankT[j] if y_rankT.numel() > 0 else torch.tensor(0., dtype=torch.float32))

        # pull per-rank targets from shard
        y_pr = m.get("y_per_rank", torch.zeros((0,), dtype=torch.float32))
        if y_pr.ndim == 2 and y_pr.shape[0] > j:
            y_per_rank_full = y_pr[j]
        else:
            y_per_rank_full = torch.zeros(0, dtype=torch.float32)

        # names, if present
        rank_names = CANONICAL_RANKS

        # *** normalize targets to our 7 heads (drop strain; collapse domain/kingdom) ***
        y_per_rank7 = normalize_y_per_rank_to7(y_per_rank_full, rank_names)

        # remap rank_index for gating/eval to our 0..6 space (or -1 if not applicable)
        ri = m.get("rank_index", torch.full((1,), -1, dtype=torch.int8))
        raw_rix = int(ri[j].item() if isinstance(ri, torch.Tensor) else ri[j])
        rank_index7 = remap_rank_index_to7(raw_rix, rank_names)

        L = int(m["lengths"][j].item() if "lengths" in m and isinstance(m["lengths"], torch.Tensor)
                else (m["lengths"][j] if "lengths" in m else x.size(-1)))

        # --- Add seq_id and taxon if present ---
        seq_id = None
        if "seq_id" in m:
            val = m["seq_id"][j]
            seq_id = val if isinstance(val, str) else str(val)
        taxon = None
        if "taxon" in m:
            val = m["taxon"][j]
            taxon = val if isinstance(val, str) else str(val)

        return {
            "x": x,
            "y_any": float(y_any.item()),
            "y_rank": float(y_rank.item()),
            "y_per_rank": y_per_rank7.float(),   # <-- exactly [7]
            "rank_index": rank_index7,           # <-- in {0..6} or -1
            "length": L,
            "seq_id": seq_id,
            "taxon": taxon,
        }


def make_loader(ds, batch_size, train=True, num_workers=2, shuffle=True, crop_max=CROP_MAX_T):
    """
    Create a DataLoader for a ShardedCFTorchDataset with custom batching and collation

    Args:
        ds (ShardedCFTorchDataset): Dataset instance
        batch_size (int): Batch size
        train (bool, optional): If True, enables random cropping and shuffling. Defaults to True
        num_workers (int, optional): Number of worker processes. Defaults to 2
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True
        crop_max (int, optional): Maximum sequence length for cropping/padding. Defaults to CROP_MAX_T

    Returns:
        DataLoader: Configured DataLoader instance.
    """
    use_mp = num_workers > 0
    sampler = ShardBatchSampler(ds, batch_size=batch_size, shuffle=shuffle if train else False)
    return DataLoader(
        ds,
        batch_sampler=sampler,                         # grouped by shard
        collate_fn=PadMaskCollateCF(train=train, max_len=crop_max),
        num_workers=num_workers,                       # keep small (<=2) to avoid N× shard dup
        pin_memory=False,                              # avoid extra pinned copies on CPU
        persistent_workers=use_mp,                     # reuse per-worker LRU
        prefetch_factor=(2 if use_mp else None),       # small to cap in-flight batches
        multiprocessing_context=("forkserver" if use_mp else None),
    )


def build_loader(args, input_path, batch_size, train_flag, rank_filter=None):
    """
    Build a ShardedCFTorchDataset and DataLoader, optionally filtering by rank

    Args:
        args (argparse.Namespace): Arguments with dataset and loader options
        input_path (str or Path): Path to shard directory or manifest
        batch_size (int): Batch size
        train_flag (bool): If True, enables training mode (shuffling, cropping)
        rank_filter (str or None, optional): Rank name to filter samples by. Defaults to None

    Returns:
        tuple: (ShardedCFTorchDataset, DataLoader)
    """
    downcast = None if args.downcast == "none" else "float16"
    if rank_filter is None:
        ds = ShardedCFTorchDataset(
            input_path, subset_index=None,
            cache_shards=args.cache_shards,
            to_float32=args.cpu_float32,
            downcast_cache_dtype=downcast
        )
    else:
        idx, _stats = build_rank_filtered_index(input_path, rank_filter, cache_file=args.rank_cache)
        if len(idx) == 0:
            raise SystemExit(f"No samples for predicted rank '{rank_filter}' in {input_path}.")
        ds = ShardedCFTorchDataset(
            input_path, subset_index=idx,
            cache_shards=args.cache_shards,
            to_float32=args.cpu_float32,
            downcast_cache_dtype=downcast
        )
    ld = make_loader(ds, batch_size=batch_size, shuffle=True, train=train_flag,
                        num_workers=args.num_workers, crop_max=(args.crop_max if train_flag else None))
    return ds, ld
import os
import time
import gc
import json
import logging
import bisect
import numpy as np
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
    def __init__(self, manifest_or_dir, split_dir=None, split="train",
                 cache_shards=1, to_float32=False, downcast_cache_dtype="float16"):
        p = Path(manifest_or_dir).resolve()
        
        self.split_dir = Path(split_dir).resolve() if split_dir else None
        self.split = split  # "train" or "val"
    
        # --- load shard paths (absolute) ---
        if p.is_dir():
            self.paths = [str(x.resolve()) for x in sorted(p.glob("*.pt"))]
            self.manifest_dir = p
            mani_sizes = None
        elif p.suffix.lower() == ".json":
            mani = json.loads(p.read_text())
            base = p.parent
            outs = mani.get("outputs", [])
            if not outs:
                raise FileNotFoundError(f"manifest has no outputs: {p}")
            # self.paths = [str((Path(x) if os.path.isabs(x) else (base / x)).resolve()) for x in outs]
            # self.paths = [x if "/home/mnguye99/scratch/training_data_inclusion_exclusion/runs/" in x else os.path.join('/home/mnguye99/scratch/training_data_inclusion_exclusion/runs/', x) for x in outs]
            self.paths = [os.path.join(base, x) for x in outs]
            self.manifest_dir = base
            mani_sizes = mani.get("sizes", None)
        else:
            raise ValueError("Pass a shard directory or permute_manifest.json")

        # --- sizes: prefer split_dir/sizes.json, else manifest sizes, else fallback ---
        self.sizes = None
        if self.split_dir and (self.split_dir / "sizes.json").exists():
            sj = json.loads((self.split_dir / "sizes.json").read_text())
            self.sizes = [int(x) for x in sj["sizes"]]
        elif mani_sizes is not None:
            self.sizes = [int(x) for x in mani_sizes]
        else:
            # expensive fallback
            self.sizes = []
            for sp in self.paths:
                m = torch.load(sp, map_location="cpu")
                n = int(m["x"].shape[0]) if "x" in m else len(m["x_list"])
                self.sizes.append(n)
                del m

        if len(self.sizes) != len(self.paths):
            raise ValueError(f"sizes length {len(self.sizes)} != #paths {len(self.paths)}")

        # --- prefix sums for global indexing ---
        self.offsets = [0]
        for n in self.sizes:
            self.offsets.append(self.offsets[-1] + n)
        self.total = self.offsets[-1]

        # shard cache
        self._cache = OrderedDict()
        self._cache_cap = max(1, int(cache_shards))
        self.to_float32 = bool(to_float32)
        self.downcast_cache_dtype = downcast_cache_dtype

        # cache allowed local idx per shard (small)
        self._allowed = {}

        logger.info("Dataset: %d shards, %d samples (split=%s)",
                    len(self.paths), self.total, self.split)
        
    def __len__(self): 
        """
        Returns the number of samples in the dataset

        Returns:
            int: Number of samples
        """
        # return len(self.index)
        return self.total
    
    def _locate(self, i: int):
        si = bisect.bisect_right(self.offsets, i) - 1
        j = i - self.offsets[si]
        return si, j
    
    def allowed_local_indices(self, si: int) -> np.ndarray:
        """
        Local indices allowed under the split. Used by sampler.
        """
        if self.split_dir is None:
            return np.arange(self.sizes[si], dtype=np.int32)

        if si in self._allowed:
            return self._allowed[si]

        mask_path = self.split_dir / f"valmask_{si:06d}.pt"
        valmask = torch.load(mask_path, map_location="cpu")

        mask = valmask if self.split == "val" else ~valmask
        idx = torch.nonzero(mask, as_tuple=False).view(-1).to(torch.int32).cpu().numpy()
        self._allowed[si] = idx
        return idx

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

    def _get_shard(self, si: int):
        if si in self._cache:
            self._cache.move_to_end(si)
            return self._cache[si]
        m = torch.load(self.paths[si], map_location="cpu")
        if self.downcast_cache_dtype == "float16":
            if "x" in m and isinstance(m["x"], torch.Tensor) and m["x"].dtype == torch.float32:
                m["x"] = m["x"].half()
        self._cache[si] = m
        if len(self._cache) > self._cache_cap:
            _, ev = self._cache.popitem(last=False)
            for k in ("x", "x_list"):
                if k in ev:
                    ev[k] = None
            del ev
        return m
    
    def __getitem__(self, i):
        si, j = self._locate(i)
        m = self._get_shard(si)

        x = (m["x"][j] if "x" in m else m["x_list"][j])
        if self.to_float32 and x.dtype != torch.float32:
            x = x.float()

        # y_any = m.get("y_any", m.get("y"))[j].float()

        # y_rankT = m.get("y_rank", torch.zeros((), dtype=torch.float32)).reshape(-1)
        # y_rank = (y_rankT[j] if y_rankT.numel() > 0 else torch.tensor(0., dtype=torch.float32))

        y_pr = m.get("y_per_rank", torch.zeros((0,), dtype=torch.int8))
        y_per_rank_full = y_pr[j] if (y_pr.ndim == 2 and y_pr.shape[0] > j) else torch.zeros(0)

        y_per_rank7 = normalize_y_per_rank_to7(y_per_rank_full, CANONICAL_RANKS)

        ri = m.get("rank_index", torch.full((1,), -1, dtype=torch.int8))
        raw_rix = int(ri[j].item() if isinstance(ri, torch.Tensor) else ri[j])
        rank_index7 = remap_rank_index_to7(raw_rix, CANONICAL_RANKS)

        L = int(m["lengths"][j].item() if "lengths" in m and isinstance(m["lengths"], torch.Tensor)
                else (m["lengths"][j] if "lengths" in m else x.size(-1)))

        seq_id = str(m["seq_id"][j]) if "seq_id" in m else None
        taxon = str(m["taxon"][j]) if "taxon" in m else None

        return {
            "x": x,
            # "y_any": float(y_any.item()),
            # "y_rank": float(y_rank.item()),
            "y_per_rank": y_per_rank7,
            "rank_index": rank_index7,
            "length": L,
            "seq_id": seq_id,
            "taxon": taxon,
        }



def make_loader(args, ds, batch_size, train=True, val=False, num_workers=2, shuffle=True, crop_max=CROP_MAX_T):
    """
    Create a DataLoader for a ShardedCFTorchDataset with custom batching and collation

    Args:
        ds (ShardedCFTorchDataset): Dataset instance
        batch_size (int): Batch size
        train (bool, optional): If True, enables random cropping and shuffling. Defaults to True
        val (bool, optional): If True, enables validation mode. Defaults to False
        num_workers (int, optional): Number of worker processes. Defaults to 2
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True
        crop_max (int, optional): Maximum sequence length for cropping/padding. Defaults to CROP_MAX_T

    Returns:
        DataLoader: Configured DataLoader instance.
    """
    use_mp = num_workers > 0
    sampler = ShardBatchSampler(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle if train else False,
        seed=args.seed,
        shards_per_epoch=args.shards_per_epoch if train else None,
        samples_per_shard=args.samples_per_shard if train else None,
        val_shards_per_epoch=args.val_shards_per_epoch if val else None,
        val_samples_per_shard=args.val_samples_per_shard if val else None,
    )
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


def build_loader(args, input_path, batch_size, train_flag, val_flag, rank_filter=None):
    """
    Build a ShardedCFTorchDataset and DataLoader, optionally filtering by rank

    Args:
        args (argparse.Namespace): Arguments with dataset and loader options
        input_path (str or Path): Path to shard directory or manifest
        batch_size (int): Batch size
        train_flag (bool): If True, enables training mode (shuffling, cropping)
        val_flag (bool): If True, enables validation mode
        rank_filter (str or None, optional): Rank name to filter samples by. Defaults to None

    Returns:
        tuple: (ShardedCFTorchDataset, DataLoader)
    """
    downcast = None if args.downcast == "none" else "float16"

    if rank_filter is None:
        ds = ShardedCFTorchDataset(
            input_path,
            split_dir=args.split_dir,
            split="train" if train_flag else "val",
            cache_shards=args.cache_shards,
            to_float32=args.cpu_float32,
            downcast_cache_dtype=downcast
        )
    else:
        idx, _stats = build_rank_filtered_index(input_path, rank_filter, cache_file=args.rank_cache)
        if len(idx) == 0:
            raise SystemExit(f"No samples for predicted rank '{rank_filter}' in {input_path}.")
        ds = ShardedCFTorchDataset(
            input_path,
            split_dir=args.split_dir,
            split="train" if train_flag else "val",
            cache_shards=args.cache_shards,
            to_float32=args.cpu_float32,
            downcast_cache_dtype=downcast
        )

    # IMPORTANT: keep crop_max fixed for BOTH train and val for deterministic shapes
    crop_max = args.crop_max if getattr(args, "crop_max", None) is not None else CROP_MAX_T

    ld = make_loader(
        args,
        ds,
        batch_size=batch_size,
        shuffle=True,                 # make_loader will disable shuffle if train_flag=False
        train=train_flag,
        val=val_flag,
        num_workers=args.num_workers,
        crop_max=crop_max,
    )
    return ds, ld
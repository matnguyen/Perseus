#!/usr/bin/env python3
import os, gc, json, argparse, time, math
from pathlib import Path
from collections import OrderedDict, defaultdict
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from alive_progress import alive_bar

import torch.multiprocessing as mp
try: mp.set_start_method("forkserver", force=True)
except RuntimeError: pass
try: mp.set_sharing_strategy("file_system")
except RuntimeError: pass

# -------------------------
# Logging
# -------------------------
LOG = logging.getLogger("train_by_rank")

def setup_logging(level: str = "INFO"):
    level = level.upper()
    if level not in ("DEBUG","INFO","WARNING","ERROR","CRITICAL"):
        level = "INFO"
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

# Limit BLAS thread fan-out (helps RAM too)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# -------------------------
# Config
# -------------------------
N_CHANNELS = 22
CROP_MAX_T = 4096

CANONICAL_RANKS = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
HEADS7_SET = set(CANONICAL_RANKS)
RANK_TO_IDX = {r:i for i,r in enumerate(CANONICAL_RANKS)}
TARGET_RANKS = CANONICAL_RANKS

def canonicalize_rank(rank: str | None) -> str | None:
    if not rank: return None
    rank = rank.lower()
    return "superkingdom" if rank in ("kingdom","domain") else rank

def normalize_y_per_rank_to7(y_per_rank: torch.Tensor,
                             rank_names: list | None) -> torch.Tensor:
    """
    Map arbitrary per-rank targets onto HEADS7 in that order.
    Unknown/missing heads -> -1 (ignored by masked loss).
    If both 'domain' and 'superkingdom' exist, we take 'superkingdom'.
    """
    R = len(CANONICAL_RANKS)
    out = torch.full((R,), -1.0, dtype=y_per_rank.dtype, device=y_per_rank.device)

    if rank_names is not None and len(rank_names) == y_per_rank.shape[-1]:
        name_to_col = {}
        for j, nm in enumerate(rank_names):
            cj = canonicalize_rank(str(nm))
            if cj is None: 
                continue
            # Keep first seen canonical column; if both domain & superkingdom exist, the latter overwrites
            if cj in HEADS7_SET:
                name_to_col[cj] = j
        for k, rk in enumerate(CANONICAL_RANKS):
            j = name_to_col.get(rk, None)
            if j is not None:
                out[k] = y_per_rank[j]
        return out

    # Fallback heuristics without rank_names
    R_src = y_per_rank.shape[-1]
    if R_src == len(CANONICAL_RANKS):          # already 7 → assume correct order
        return y_per_rank
    if R_src > len(CANONICAL_RANKS):           
        return y_per_rank[1:8]
    # Too few → pad unknowns (-1)
    pad = torch.full((len(CANONICAL_RANKS)-R_src,), -1.0, dtype=y_per_rank.dtype, device=y_per_rank.device)
    return torch.cat([y_per_rank, pad], dim=0)

def normalize_y_per_rank_to9(y_per_rank: torch.Tensor,
                             rank_names: list | None) -> torch.Tensor:
    R = len(CANONICAL_RANKS)
    if y_per_rank.shape[-1] == R:
        return y_per_rank
    # If input is shorter, pad with -1
    pad = torch.full((R - y_per_rank.shape[-1],), -1.0, dtype=y_per_rank.dtype, device=y_per_rank.device)
    return torch.cat([y_per_rank, pad], dim=0)

def remap_rank_index_to7(rank_ix_raw: int,
                         rank_names: list | None) -> int:
    """
    Convert a shard's rank_index (over its own rank set) to our 0..6 over HEADS7.
    Returns -1 if the raw rank corresponds to strain/subspecies/etc. not in HEADS7.
    """
    if rank_names is None or rank_ix_raw < 0 or rank_ix_raw >= len(rank_names):
        return -1
    name = canonicalize_rank(str(rank_names[rank_ix_raw]))
    if name in HEADS7_SET:
        return CANONICAL_RANKS.index(name)
    return -1

def remap_rank_index_to9(rank_ix_raw: int,
                         rank_names: list | None) -> int:
    R = len(CANONICAL_RANKS)
    if rank_names is None or rank_ix_raw < 0 or rank_ix_raw >= len(rank_names):
        return -1
    name = canonicalize_rank(str(rank_names[rank_ix_raw]))
    if name in HEADS7_SET:
        return CANONICAL_RANKS.index(name)
    return -1

# -------------------------
# Dataset (channel-first) with Option-B labels
#   Optimizations:
#   - Per-worker tiny LRU shard cache (cache_shards=1 by default)
#   - Optional fp16 downcast in the cache to halve RAM
#   - Aggressive eviction (break references + gc.collect)
# -------------------------
class ShardedCFTorchDataset(Dataset):
    """
    Loads channel-first shards (.pt) that include:
      x:[N,C,T], y (legacy mirror of y_any), y_any:[N], y_rank:[N],
      y_per_rank:[N,R] (optional), rank_index:[N], seq_id:[N], taxon:[N], lengths:[N]?.
    If subset_index is provided, it is a list of (shard_idx, local_idx).
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
                    m = torch.load(path, map_location="cpu")
                    n = int(m["x"].shape[0])
                    self.index.extend((si, j) for j in range(n))
        LOG.info("Dataset: %d shards, %d samples (cache_shards=%d, downcast=%s)",
                 len(self.paths), len(self.index), self._cache_cap, self.downcast_cache_dtype or "none")

    def __len__(self): return len(self.index)

    def _downcast_inplace(self, m):
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
        evict_si, evict_m = self._cache.popitem(last=False)
        # Break references for large tensors so storage is released
        for k in ("x","x_list"):
            if k in evict_m:
                evict_m[k] = None
        del evict_m
        gc.collect()

    def _get_shard(self, si: int):
        if si in self._cache:
            self._cache.move_to_end(si)
            return self._cache[si]
        t0 = time.perf_counter()
        m = torch.load(self.paths[si], map_location="cpu")
        if self.downcast_cache_dtype:
            m = self._downcast_inplace(m)
        self._cache[si] = m
        if len(self._cache) > self._cache_cap:
            self._evict_one()
        LOG.debug("Shard cache: loaded si=%d in %.3fs", si, time.perf_counter() - t0)
        return m
    
    def __getitem__(self, i):
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
        # rank_names = m.get("rank_names", None)
        rank_names = CANONICAL_RANKS

        # # *** normalize targets to our 7 heads (drop strain; collapse domain/kingdom) ***
        # y_per_rank7 = normalize_y_per_rank_to7(y_per_rank_full, rank_names)
        y_per_rank9 = normalize_y_per_rank_to9(y_per_rank_full, rank_names)

        # remap rank_index for gating/eval to our 0..6 space (or -1 if not applicable)
        ri = m.get("rank_index", torch.full((1,), -1, dtype=torch.int8))
        raw_rix = int(ri[j].item() if isinstance(ri, torch.Tensor) else ri[j])
        # rank_index7 = remap_rank_index_to7(raw_rix, rank_names)
        rank_index9 = remap_rank_index_to9(raw_rix, rank_names)

        L = int(m["lengths"][j].item() if "lengths" in m and isinstance(m["lengths"], torch.Tensor)
                else (m["lengths"][j] if "lengths" in m else x.size(-1)))

        return {
            "x": x,
            "y_any": float(y_any.item()),
            "y_rank": float(y_rank.item()),
            # "y_per_rank": y_per_rank7.float(),   # <-- exactly [7]
            "y_per_rank": y_per_rank9.float(),     # <-- exactly [9]
            # "rank_index": rank_index7,           # <-- in {0..6} or -1
            "rank_index": rank_index9,             # <-- in {0..8} or -1
            "length": L,
        }

    # def __getitem__(self, i):
    #     si, j = self.index[i]
    #     m = self._get_shard(si)

    #     # pick from "x" or "x_list" depending how shard was saved
    #     x = (m["x"][j] if "x" in m else m["x_list"][j])  # [C,T]

    #     # Optional training-time crop here if you want fixed length in-stack.
    #     # If you prefer variable with pad+mask, skip this and let collate pad.
    #     # (We keep variable + pad below.)
    #     if self.to_float32 and x.dtype != torch.float32:
    #         x = x.float()

    #     # Labels (with fallbacks)
    #     y_any   = m.get("y_any", m.get("y"))[j].float()
    #     y_rankT = m.get("y_rank", torch.zeros((), dtype=torch.float32)).reshape(-1)
    #     y_rank  = (y_rankT[j] if y_rankT.numel() > 0 else torch.tensor(0., dtype=torch.float32))
    #     y_pr    = m.get("y_per_rank", torch.zeros((0,), dtype=torch.float32))
    #     if y_pr.ndim == 2 and y_pr.shape[0] > j:
    #         y_per_rank = y_pr[j]
    #     else:
    #         y_per_rank = torch.zeros(len(CANONICAL_RANKS), dtype=torch.float32)

    #     ri      = m.get("rank_index", torch.full((1,), -1, dtype=torch.int8))
    #     rank_ix = int(ri[j].item() if isinstance(ri, torch.Tensor) else ri[j])

    #     if "lengths" in m:
    #         L = int(m["lengths"][j].item() if isinstance(m["lengths"], torch.Tensor) else m["lengths"][j])
    #     else:
    #         L = x.size(-1)

    #     return {
    #         "x": x,                                # [C,T]
    #         "y_any": float(y_any.item()),
    #         "y_rank": float(y_rank.item()),
    #         "y_per_rank": y_per_rank.float(),      # [R]
    #         "rank_index": rank_ix,                 # int
    #         "length": L,
    #         "shard": si,                           # for batch-by-shard sampler
    #     }

# -------------------------
# Collates (pad + mask)
#   Keep variable-length by padding to batch max; mask for pooling.
# -------------------------
class PadMaskCollateCF:
    def __init__(self, max_len=CROP_MAX_T, train=True):
        self.max_len = max_len if train else None
        self.train   = train

    def __call__(self, batch):
        xs = [b["x"] for b in batch]            # [C,Ti]
        proc, lens = [], []
        for x in xs:
            T = x.size(-1)
            if self.max_len and T > self.max_len:
                st = torch.randint(0, T - self.max_len + 1, (1,)).item()
                x = x[..., st:st + self.max_len]
                T = x.size(-1)
            proc.append(x); lens.append(T)
        T_max = max(lens); B, C = len(proc), proc[0].size(0)
        X  = torch.zeros(B, C, T_max, dtype=proc[0].dtype)
        M  = torch.zeros(B, 1, T_max, dtype=torch.bool)
        for i, x in enumerate(proc):
            Ti = x.size(-1)
            X[i, :, :Ti] = x
            M[i, 0, :Ti] = True

        # labels
        y_any  = torch.tensor([b["y_any"] for b in batch], dtype=torch.float32)
        y_rank = torch.tensor([b["y_rank"] for b in batch], dtype=torch.float32)
        y_pr   = torch.stack([b["y_per_rank"] for b in batch], dim=0)
        rix    = torch.tensor([b["rank_index"] for b in batch], dtype=torch.int64)
        Ls     = torch.tensor(lens, dtype=torch.int32)

        return {"x": X, "mask": M, "lengths": Ls, "y_any": y_any, "y_rank": y_rank,
                "y_per_rank": y_pr, "rank_index": rix}

# -------------------------
# Shard-grouped BatchSampler
#   Ensures each batch pulls from a single shard → cache reuse, lower RAM.
# -------------------------
class ShardBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: ShardedCFTorchDataset, batch_size: int,
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
        n = 0
        for idxs in self.per_shard.values():
            n += (len(idxs) // self.bs) if self.drop_last else math.ceil(len(idxs) / self.bs)
        return n

def make_loader(ds, batch_size, train=True, num_workers=2, shuffle=True, crop_max=CROP_MAX_T):
    use_mp = num_workers > 0
    sampler = ShardBatchSampler(ds, batch_size=batch_size, shuffle=shuffle if train else False,
                                drop_last=train)
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

# -------------------------
# Models (single or multi-head)
# -------------------------
def masked_avgpool1d(h: torch.Tensor, m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if m.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        m = m.float()
    num = (h * m).sum(-1)
    den = m.sum(-1).clamp_min(eps)
    return num / den

class CNN1D_CF(nn.Module):
    def __init__(self, in_channels=N_CHANNELS, out_dim=1, extra_dim=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + extra_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x, mask=None, extra=None):   # x:[B,C,T]
        h = self.conv(x)                            # [B,256,T]
        h_vec = masked_avgpool1d(h, mask) if mask is not None else F.adaptive_avg_pool1d(h,1).squeeze(-1)
        if extra is not None:
            h_vec = torch.cat([h_vec, extra], dim=1)
        return self.classifier(h_vec)               # [B,out_dim]

class Bottleneck1D(nn.Module):
    def __init__(self, c_in, c_out, dilation=1, stride=1):
        super().__init__()
        mid = max(32, c_out // 4)
        self.proj = (nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False)
                     if (c_in != c_out or stride != 1) else nn.Identity())
        self.net = nn.Sequential(
            nn.Conv1d(c_in, mid, 1, stride=stride, bias=False),
            nn.BatchNorm1d(mid), nn.GELU(),
            nn.Conv1d(mid, mid, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(mid), nn.GELU(),
            nn.Conv1d(mid, c_out, 1, bias=False),
            nn.BatchNorm1d(c_out),
        )
    def forward(self, x):
        y = self.net(x); r = self.proj(x)
        if y.size(-1) != r.size(-1):
            T = min(y.size(-1), r.size(-1))
            y, r = y[..., :T], r[..., :T]
        return F.gelu(y + r)

class ResTCN_CF(nn.Module):
    def __init__(self, in_channels=N_CHANNELS, out_dim=1, extra_dim=1, widths=(64,128,256), dilations=(1,2,4,8)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, widths[0], 7, padding=3),
                                  nn.BatchNorm1d(widths[0]), nn.GELU())
        blocks = []; c = widths[0]
        for w in widths:
            blocks += [Bottleneck1D(c, w, dilation=dilations[0], stride=1 if c==w else 2),
                       Bottleneck1D(w, w, dilation=dilations[1], stride=1)]
            c = w
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(c + extra_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, out_dim))

    def forward(self, x, mask=None, extra=None):
        h = self.body(self.stem(x))
        h_vec = masked_avgpool1d(h, mask) if mask is not None else F.adaptive_avg_pool1d(h,1).squeeze(-1)
        if extra is not None:
            h_vec = torch.cat([h_vec, extra], dim=1)
        return self.head(h_vec)  # [B,out_dim]

# -------------------------
# Losses
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
    def forward(self, logits, targets, mask=None):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p   = torch.sigmoid(logits)
        pt  = p * targets + (1 - p) * (1 - targets)
        loss = self.alpha * ((1 - pt) ** self.gamma) * bce
        if mask is not None:
            loss = loss * mask
        if self.reduction == "mean":
            denom = mask.sum().clamp_min(1.0) if mask is not None else torch.tensor(loss.numel(), device=loss.device, dtype=loss.dtype)
            return loss.sum() / denom
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

# -------------------------
# Train / Validate
# -------------------------
def compute_loss_from_batch(logits, batch, device, crit, target_mode, rank_idx_for_gate):
    if target_mode == "any":
        y = batch["y_any"].to(device).view(-1, 1 if logits.ndim==2 else 1).squeeze(-1)
        return crit(logits.view_as(y), y)

    if target_mode == "rank":
        y = batch["y_rank"].to(device).view(-1, 1 if logits.ndim==2 else 1).squeeze(-1)
        if rank_idx_for_gate is not None:
            gate = (batch["rank_index"] == int(rank_idx_for_gate)).to(device).float()
            return crit(logits.view_as(y), y, mask=gate)
        else:
            return crit(logits.view_as(y), y)

    # per-rank
    y = batch["y_per_rank"].to(device)               # [B,R]
    if logits.ndim == 1:
        raise ValueError("per-rank target requires model out_dim == R")
    mask = torch.ones_like(y, device=device)         # all heads valid
    return crit(logits, y, mask=mask)

@torch.no_grad()
def evaluate(model, loader, device, target_mode="any", rank_idx_for_gate=None):
    model.eval()
    crit = FocalLoss(alpha=1, gamma=2)

    total_loss = 0.0
    total_n = 0
    all_logits = []
    all_targets = []

    with alive_bar(len(loader), title="Evaluating", force_tty=True) as bar:
        for batch in loader:
            x   = batch["x"].to(device, non_blocking=True)
            msk = batch["mask"].to(device, non_blocking=True)
            extra = torch.log1p(batch["lengths"].to(device).float()).unsqueeze(1)
            
            # force tensors to fp32 on device
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            # y = y.to(device, non_blocking=True)
            if msk is not None:   msk = msk.to(device, non_blocking=True)           # bool/int ok
            if extra is not None: extra = extra.to(device, dtype=torch.float32, non_blocking=True)

            
            # if device.type == "cuda":
            #     with torch.amp.autocast('cuda', dtype=torch.float16):
            #         logits = model(x, mask=msk, extra=extra)
            #         loss = compute_loss_from_batch(logits, batch, device, crit, target_mode, rank_idx_for_gate)
            # else:
            logits = model(x.float(), mask=msk, extra=extra)
            loss = compute_loss_from_batch(logits, batch, device, crit, target_mode, rank_idx_for_gate)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_n += bs

            if target_mode in ("any","rank"):
                if target_mode == "any":
                    y = batch["y_any"].to(device)
                else:
                    y = batch["y_rank"].to(device)
                    if rank_idx_for_gate is not None:
                        gate = (batch["rank_index"] == int(rank_idx_for_gate)).to(device)
                        y = y[gate]; logits = logits[gate]
                        bs = y.numel()
                        if bs == 0:
                            bar()
                            continue
                all_targets.append(y.detach().cpu())
                all_logits.append(logits.detach().cpu())
            bar()

    metrics = {"loss": total_loss / max(total_n,1)}
    if target_mode in ("any","rank") and all_targets:
        y = torch.cat(all_targets).float().numpy()
        s = torch.sigmoid(torch.cat(all_logits)).numpy()
        pred = (s >= 0.5).astype(np.int32)
        acc = (pred == y.astype(np.int32)).mean() if y.size else 0.0
        auroc = _binary_auroc(y, s)
        metrics.update({"acc": float(acc), "auroc": float(auroc)})
    return metrics

def _binary_auroc(y_true, y_score):
    y_true = y_true.astype(np.int32)
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    p = y.sum()
    n = len(y) - p
    if p == 0 or n == 0:
        return 0.5
    tp = 0; fp = 0
    tps = [0]; fps = [0]
    for yi in y:
        if yi == 1: tp += 1
        else: fp += 1
        tps.append(tp); fps.append(fp)
    tps = np.array(tps, dtype=np.float64); fps = np.array(fps, dtype=np.float64)
    tpr = tps / p; fpr = fps / n
    return np.trapz(tpr, fpr)

def train(model, train_loader, val_loader, device, target_mode="any", rank_idx_for_gate=None,
          epochs=10, lr=1e-3, save_path="model_cf.pt"):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = FocalLoss(alpha=1, gamma=2)
    scaler = None
    # scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
    torch.backends.cudnn.benchmark = True

    best_metric = -1.0
    best_state = None

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for batch in train_loader:
            x   = batch["x"].to(device, non_blocking=True)
            msk = batch["mask"].to(device, non_blocking=True)
            extra = torch.log1p(batch["lengths"].to(device).float()).unsqueeze(1)
            
            # force tensors to fp32 on device
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            # y = y.to(device, non_blocking=True)
            if msk is not None:   msk = msk.to(device, non_blocking=True)           # bool/int ok
            if extra is not None: extra = extra.to(device, dtype=torch.float32, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            if scaler:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(x, mask=msk, extra=extra)
                    loss = compute_loss_from_batch(logits, batch, device, crit, target_mode, rank_idx_for_gate)
                scaler.scale(loss).backward()
                scaler.step(optim); scaler.update()
            else:
                logits = model(x, mask=msk, extra=extra)
                loss = compute_loss_from_batch(logits, batch, device, crit, target_mode, rank_idx_for_gate)
                loss.backward(); optim.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_n += bs

        LOG.info(f"Epoch {ep:02d} training complete, validating ...")
        train_loss = total_loss / max(total_n,1)

        val_metrics = evaluate(model, val_loader, device, target_mode, rank_idx_for_gate)
        LOG.info(f"Epoch {ep:02d} | train_loss={train_loss:.4f} | "
                 f"val_loss={val_metrics['loss']:.4f}" +
                 (f" | val_acc={val_metrics.get('acc',float('nan')):.4f} | val_auroc={val_metrics.get('auroc',float('nan')):.4f}"
                  if 'acc' in val_metrics else ""))

        score = (val_metrics.get("auroc", None))
        if score is None:
            score = -val_metrics["loss"]
        if score > best_metric:
            best_metric = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
            LOG.info(f"[saved best] {save_path}")
            
        torch.cuda.empty_cache(); gc.collect()

    if best_state is None:
        torch.save(model.state_dict(), save_path)
        LOG.info(f"[saved last] {save_path}")

# -------------------------
# Rank indexing (optional filter)
# -------------------------
def build_rank_filtered_index(shard_dir_or_manifest: str, target_rank: str,
                              cache_file: str | None = None) -> tuple[list[tuple[int,int]], dict]:
    p = Path(shard_dir_or_manifest)
    if p.is_dir():
        shard_paths = sorted(str(x) for x in p.glob("*.pt"))
        root_dir = p
        sizes = None
    else:
        mani = json.loads(p.read_text())
        shard_paths = mani.get("outputs", [])
        root_dir = p.parent
        sizes = mani.get("sizes", None)
    if not shard_paths:
        raise FileNotFoundError("No shard .pt files found.")

    if cache_file is None:
        cache_file = str(root_dir / "rank_index_cache.json")

    by_rank = defaultdict(list)
    stats = defaultdict(int)

    if sizes is not None and "rank_index_offsets" in mani:
        # Optional fast path if manifest carries rank info; otherwise fall back to scan
        pass

    for si, sp in enumerate(shard_paths):
        m = torch.load(sp, map_location="cpu")
        ri = m.get("rank_index", None)
        n  = int(m["x"].shape[0])
        if ri is None:
            for j in range(n):
                by_rank[target_rank].append((si, j))
                stats[target_rank] += 1
        else:
            for j in range(n):
                idx = int(ri[j].item() if isinstance(ri, torch.Tensor) else ri[j])
                rk = CANONICAL_RANKS[idx] if 0 <= idx < len(CANONICAL_RANKS) else None
                if rk: stats[rk] += 1
                if rk == target_rank:
                    by_rank[target_rank].append((si, j))
        del m
        if (si + 1) % 10 == 0:
            gc.collect()

    payload = {"by_rank": {k:v for k,v in by_rank.items()}, "stats": dict(stats)}
    try:
        Path(cache_file).write_text(json.dumps(payload))
    except Exception:
        pass
    return by_rank.get(target_rank, []), dict(stats)


def build_loader(input_path, batch_size, train_flag, rank_filter=None):
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Train from (train,val) shard manifests with Option-B labels (optimized I/O).")
    ap.add_argument("--train", required=False, help="Train shard directory OR train_manifest.json")
    ap.add_argument("--val",   required=False, help="Val shard directory OR val_manifest.json")
    ap.add_argument("--input", required=False, help="Shard directory OR single manifest (used for both if --val missing)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)  # <=2 to avoid N× memory
    ap.add_argument("--model", choices=["cnn","restcn"], default="cnn")
    ap.add_argument("--save",   default="model_cf.pt")

    ap.add_argument("--target", choices=["any","rank","per-rank"], default="any")

    group = ap.add_mutually_exclusive_group()
    group.add_argument("--rank", choices=TARGET_RANKS, help="Use only samples with this predicted canonical rank")
    group.add_argument("--ranks", action="store_true", help="Train one model per predicted canonical rank (loop)")

    ap.add_argument("--rank_cache", default=None, help="Optional cache path for rank index")
    ap.add_argument("--log-level", default="INFO", help="DEBUG | INFO | WARNING | ERROR | CRITICAL")
    ap.add_argument("--crop-max", type=int, default=CROP_MAX_T, help="Max crop length for TRAIN loader (no crop for VAL)")
    # Loader/Dataset memory knobs
    ap.add_argument("--cache-shards", type=int, default=1, help="Shards kept in RAM per worker")
    ap.add_argument("--downcast", choices=["none","fp16"], default="fp16", help="Downcast shard tensors in cache")
    ap.add_argument("--cpu-float32", action="store_true", help="Cast samples to float32 on CPU before batching")

    args = ap.parse_args()
    setup_logging(args.log_level)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("Using device: %s (CUDA avail=%s)", device, torch.cuda.is_available())

    # Resolve inputs
    if args.train or args.val:
        if not args.train or not args.val:
            raise SystemExit("Please provide both --train and --val (paths to shard dirs or manifests).")
        train_input = args.train
        val_input   = args.val
    else:
        if not args.input:
            raise SystemExit("Provide either --train and --val, or a single --input.")
        train_input = val_input = args.input
        LOG.warning("--train/--val not provided; using --input for both.")

    def make_model(out_dim):
        if args.model == "cnn":
            LOG.info("Model: CNN1D_CF (out_dim=%d)", out_dim)
            return CNN1D_CF(in_channels=N_CHANNELS, out_dim=out_dim, extra_dim=1).to(device)
        else:
            LOG.info("Model: ResTCN_CF (out_dim=%d)", out_dim)
            return ResTCN_CF(in_channels=N_CHANNELS, out_dim=out_dim, extra_dim=1).to(device)

    out_dim = 1 if args.target in ("any","rank") else len(CANONICAL_RANKS)

    if args.ranks:
        for rk in TARGET_RANKS:
            LOG.info("=== Predicted Rank: %s ===", rk)
            _, train_loader = build_loader(train_input, args.batch, True,  rank_filter=rk)
            _, val_loader   = build_loader(val_input,   args.batch, False, rank_filter=rk)
            model = make_model(out_dim)
            save_path = f"{Path(args.save).with_suffix('')}_{args.target}_{rk}.pt"
            rank_idx_gate = RANK_TO_IDX[rk] if args.target == "rank" else None
            train(model, train_loader, val_loader, device,
                  target_mode=args.target, rank_idx_for_gate=rank_idx_gate,
                  epochs=args.epochs, lr=args.lr, save_path=save_path)
            torch.cuda.empty_cache(); gc.collect()

    if args.rank:
        rk = args.rank
        LOG.info("Training single model for predicted rank='%s'", rk)
        _, train_loader = build_loader(train_input, args.batch, True,  rank_filter=rk)
        _, val_loader   = build_loader(val_input,   args.batch, False, rank_filter=rk)
        model = make_model(out_dim)
        save_path = f"{Path(args.save).with_suffix('')}_{args.target}_{rk}.pt"
        rank_idx_gate = RANK_TO_IDX[rk] if args.target == "rank" else None
        train(model, train_loader, val_loader, device,
              target_mode=args.target, rank_idx_for_gate=rank_idx_gate,
              epochs=args.epochs, lr=args.lr, save_path=save_path)

    LOG.info("Training on ALL samples (no rank filter). target=%s", args.target)
    _, train_loader = build_loader(train_input, args.batch, True,  rank_filter=None)
    _, val_loader   = build_loader(val_input,   args.batch, False, rank_filter=None)
    model = make_model(out_dim)
    save_path = f"{Path(args.save).with_suffix('')}_{args.target}.pt"
    train(model, train_loader, val_loader, device,
          target_mode=args.target, rank_idx_for_gate=None,
          epochs=args.epochs, lr=args.lr, save_path=save_path)

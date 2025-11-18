import gc
import torch
import json
from collections import defaultdict
from pathlib import Path

from taxoncnn.utils.constants import CANONICAL_RANKS
from taxoncnn.utils.tax_utils import canonicalize_rank

def normalize_y_per_rank_to7(y_per_rank, rank_names):
    """
    Map arbitrary per-rank targets onto the canonical 7 ranks (HEADS7) in order

    Unknown or missing ranks are set to -1 (ignored by masked loss)
    If both 'domain' and 'superkingdom' exist, 'superkingdom' is preferred

    Args:
        y_per_rank (torch.Tensor): Tensor of per-rank targets, shape (..., R_src)
        rank_names (list or None): List of rank names corresponding to y_per_rank columns

    Returns:
        torch.Tensor: Tensor of shape (..., 7) with values mapped to canonical ranks, missing set to -1
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
            if cj in CANONICAL_RANKS:
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


def remap_rank_index_to7(rank_ix_raw, rank_names):
    """
    Convert a shard's rank_index (over its own rank set) to canonical 0..6 over HEADS7

    Returns -1 if the raw rank corresponds to a rank not in HEADS7 (e.g., strain/subspecies)

    Args:
        rank_ix_raw (int): Raw rank index from the shard
        rank_names (list or None): List of rank names for the shard

    Returns:
        int: Canonical rank index (0..6), or -1 if not found
    """
    if rank_names is None or rank_ix_raw < 0 or rank_ix_raw >= len(rank_names):
        return -1
    name = canonicalize_rank(str(rank_names[rank_ix_raw]))
    if name in CANONICAL_RANKS:
        return CANONICAL_RANKS.index(name)
    return -1


def build_rank_filtered_index(shard_dir_or_manifest, target_rank, cache_file):
    """
    Build an index of (shard_idx, local_idx) pairs for samples matching a target rank

    Scans all .pt shards in a directory or manifest, and collects indices of samples whose rank matches
    the specified target_rank. Optionally caches the result to a JSON file

    Args:
        shard_dir_or_manifest (str): Path to a directory of .pt shards or a manifest .json file
        target_rank (str): Canonical rank name to filter samples by (e.g., "species")
        cache_file (str or None, optional): Path to cache the index as a JSON file. If None, defaults to "rank_index_cache.json" in the shard directory

    Returns:
        tuple:
            - list[tuple[int, int]]: List of (shard_idx, local_idx) pairs for samples with the target rank
            - dict: Statistics dictionary with counts per rank
    """
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

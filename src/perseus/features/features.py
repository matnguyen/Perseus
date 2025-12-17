import numpy as np
import torch
import logging

import perseus.utils.globals as globals
from perseus.utils.tax_utils import (
    canonicalize_rank,
    get_canonical_taxid_for_rank,
    get_taxid_rank_raw
)

logger = logging.getLogger(__name__)


def _resample_TN_to_T(x_TN: np.ndarray, T_target: int) -> np.ndarray:
    """
    Resample sequence length dimension T of input array x_TN to target T_target using linear interpolation

    Args:
        x_TN (np.ndarray): Input array of shape (T, C)
        T_target (int): Target length for dimension T

    Returns:
        np.ndarray: Resampled array of shape (T_target, C)
    """
    T, C = x_TN.shape
    if T_target <= 0 or T == T_target:
        return x_TN
    if T == 1:
        return np.repeat(x_TN, T_target, axis=0)
    src = np.linspace(0.0, 1.0, T, endpoint=True, dtype=np.float64)
    dst = np.linspace(0.0, 1.0, T_target, endpoint=True, dtype=np.float64)
    out = np.empty((T_target, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = np.interp(dst, src, x_TN[:, c].astype(np.float64)).astype(np.float32)
    return out


def _torch_dtype(name: str) -> torch.dtype:
    """
    Convert string name to torch.dtype.

    Args:
        name (str): Name of the data type

    Returns:
        torch.dtype: Corresponding torch data type
    """
    name = name.lower()
    if name in ("float16","fp16","half"): return torch.float16
    if name in ("bfloat16","bf16"):       return torch.bfloat16
    return torch.float32


def compute_bin_features(kmer_tax_counts, pred_lineage, canonical_ranks):
    """
    Compute binned features from kmer taxonomic counts and predicted lineage

    Args:
        kmer_tax_counts (dict): Kmer taxid counts
        pred_lineage (list): Predicted lineage taxids
        canonical_ranks (list): List of canonical ranks

    Returns:
        list: Feature vector
    """
    lineage_ranks = globals.NCBI.get_rank(pred_lineage)
    lineage_at_rank = {r: None for r in canonical_ranks}
    for t in pred_lineage:
        raw = lineage_ranks.get(t)
        can = canonicalize_rank(raw)
        if can in canonical_ranks and lineage_at_rank[can] is None:
            lineage_at_rank[can] = t

    in_lineage_counts  = {r: 0 for r in canonical_ranks}
    desc_counts        = {r: 0 for r in canonical_ranks}
    out_lineage_counts = {r: 0 for r in canonical_ranks}

    total = int(sum(kmer_tax_counts.values()))
    if total == 0:
        return [np.float32(0.0)] + [np.float32(0.0) for _ in range(len(canonical_ranks) * 3)]

    lineage_map   = globals._shared_lineage_map or {}
    canonical_map = globals._shared_canonical_map or {}
    descendant_map = globals._shared_descendant_map or {}

    for taxid, count in kmer_tax_counts.items():
        try:
            taxid = int(taxid)
        except Exception:
            continue

        kmer_ancestors = canonical_map.get(taxid)
        if kmer_ancestors is None:
            kmer_ancestors = get_canonical_taxid_for_rank(taxid, canonical_ranks, globals.NCBI)

        kmer_rank_raw = get_taxid_rank_raw(taxid)
        kmer_rank = canonicalize_rank(kmer_rank_raw)

        for rank in canonical_ranks:
            pred_anc = lineage_at_rank[rank]
            if pred_anc is None:
                out_lineage_counts[rank] += count
                continue

            anc = kmer_ancestors.get(rank)

            if anc == pred_anc:
                # If the k-mer sits exactly at this node -> in_lineage, else descendant
                if (taxid == pred_anc) or (kmer_rank == rank):
                    in_lineage_counts[rank] += count
                else:
                    desc_counts[rank] += count
                continue

            pred_desc = descendant_map.get(int(pred_anc))
            if anc is None and pred_desc is not None and taxid in pred_desc:
                desc_counts[rank] += count
            else:
                out_lineage_counts[rank] += count

    denom = float(total)
    vec = [kmer_tax_counts[pred_lineage[-1]] / denom if pred_lineage and pred_lineage[-1] in kmer_tax_counts else np.float32(0.0)]
    for r in canonical_ranks:
        fi = in_lineage_counts[r]  / denom
        fo = out_lineage_counts[r] / denom
        fd = desc_counts[r]        / denom
        vec.extend([np.float32(fi), np.float32(fo), np.float32(fd)])
    return vec
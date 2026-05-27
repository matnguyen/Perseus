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


def compute_bin_features(kmer_tax_counts, pred_lineage, canonical_ranks, lineage_at_rank=None):
    """
    Compute binned features from kmer taxonomic counts and predicted lineage

    Args:
        kmer_tax_counts (dict): Kmer taxid counts
        pred_lineage (list): Predicted lineage taxids
        canonical_ranks (list): List of canonical ranks
        lineage_at_rank (dict, optional): Pre-computed {rank: taxid} map for pred_lineage.
            Pass from process_chunk_iter to avoid a DB call per bin.

    Returns:
        list: Feature vector
    """
    if lineage_at_rank is None:
        lineage_ranks = globals.NCBI.get_rank(pred_lineage)
        lineage_at_rank = {r: None for r in canonical_ranks}
        for t in pred_lineage:
            raw = lineage_ranks.get(t)
            can = canonicalize_rank(raw)
            if can in canonical_ranks and lineage_at_rank[can] is None:
                lineage_at_rank[can] = t

    n_ranks = len(canonical_ranks)
    total = int(sum(kmer_tax_counts.values()))
    if total == 0:
        return [np.float32(0.0)] * (1 + n_ranks * 3)

    canonical_map = globals._shared_canonical_map or {}

    # pred_anc per rank as int array; -1 means None
    pred_anc_arr = np.array(
        [lineage_at_rank[r] if lineage_at_rank[r] is not None else -1
         for r in canonical_ranks], dtype=np.int64
    )  # (n_ranks,)

    # rank name → index, for fast kmer_rank comparison
    rank_index = {r: i for i, r in enumerate(canonical_ranks)}

    taxids = list(kmer_tax_counts.keys())
    counts = np.array([kmer_tax_counts[t] for t in taxids], dtype=np.float32)  # (n_taxa,)
    n_taxa = len(taxids)

    taxid_int    = np.empty(n_taxa, dtype=np.int64)
    anc_matrix   = np.full((n_taxa, n_ranks), -1, dtype=np.int64)  # ancestor of taxid i at rank r
    kmer_rank_idx = np.full(n_taxa, -1, dtype=np.int64)            # own-rank index of taxid i

    for i, taxid in enumerate(taxids):
        tid = int(taxid)
        taxid_int[i] = tid
        ancs = canonical_map.get(tid)
        if ancs is None:
            ancs = get_canonical_taxid_for_rank(tid, canonical_ranks, globals.NCBI)
        for ri, rank in enumerate(canonical_ranks):
            a = ancs.get(rank)
            if a is not None:
                anc_matrix[i, ri] = int(a)
        kr = canonicalize_rank(get_taxid_rank_raw(tid))
        if kr in rank_index:
            kmer_rank_idx[i] = rank_index[kr]

    # --- vectorized classification (n_taxa × n_ranks) ---
    pred_present = pred_anc_arr != -1                                                     # (n_ranks,)
    anc_match    = anc_matrix == pred_anc_arr[np.newaxis, :]                              # (n_taxa, n_ranks)
    at_node      = taxid_int[:, np.newaxis] == pred_anc_arr[np.newaxis, :]               # (n_taxa, n_ranks)
    at_rank      = kmer_rank_idx[:, np.newaxis] == np.arange(n_ranks)[np.newaxis, :]     # (n_taxa, n_ranks)

    in_mask   = pred_present & anc_match & (at_node | at_rank)   # (n_taxa, n_ranks)
    desc_mask = pred_present & anc_match & ~(at_node | at_rank)  # (n_taxa, n_ranks)
    out_mask  = ~in_mask & ~desc_mask                             # (n_taxa, n_ranks)

    # dot product: counts (n_taxa,) · masks (n_taxa, n_ranks) → (n_ranks,)
    in_lin = counts @ in_mask.astype(np.float32)
    out    = counts @ out_mask.astype(np.float32)
    desc   = counts @ desc_mask.astype(np.float32)

    denom = float(total)
    leaf  = pred_lineage[-1] if pred_lineage else None
    vec   = [np.float32(kmer_tax_counts.get(leaf, 0) / denom)]
    for ri in range(n_ranks):
        vec += [np.float32(in_lin[ri] / denom),
                np.float32(out[ri]    / denom),
                np.float32(desc[ri]   / denom)]
    return vec
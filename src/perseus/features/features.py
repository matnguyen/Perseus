import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

import perseus.utils.globals as globals
from perseus.utils.tax_utils import (
    canonicalize_rank,
    get_canonical_taxid_for_rank,
    get_taxid_rank_raw,
)

logger = logging.getLogger(__name__)


# ============================================================
# Resampling
# ============================================================

@lru_cache(maxsize=256)
def _resample_plan(T: int, T_target: int):
    positions = np.linspace(
        0.0,
        T - 1,
        T_target,
        dtype=np.float32,
    )

    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, T - 1)

    weight = (
        positions - left.astype(np.float32)
    )[:, None]

    return left, right, weight


def _resample_TN_to_T(
    x_TN: np.ndarray,
    T_target: int,
) -> np.ndarray:

    T, _ = x_TN.shape

    if T_target <= 0 or T == T_target:
        return x_TN

    if T == 1:
        return np.repeat(
            x_TN,
            T_target,
            axis=0,
        )

    x = np.asarray(
        x_TN,
        dtype=np.float32,
    )

    left, right, weight = _resample_plan(
        T,
        T_target,
    )

    return (
        x[left] * (1.0 - weight)
        + x[right] * weight
    ).astype(
        np.float32,
        copy=False,
    )


# ============================================================
# Torch helpers
# ============================================================

def _torch_dtype(name: str) -> torch.dtype:

    name = name.lower()

    if name in (
        "float16",
        "fp16",
        "half",
    ):
        return torch.float16

    if name in (
        "bfloat16",
        "bf16",
    ):
        return torch.bfloat16

    return torch.float32


# ============================================================
# Taxonomy metadata
# ============================================================

@lru_cache(maxsize=128)
def _cached_rank_helpers(
    canonical_ranks_key: tuple,
):
    rank_index = {
        rank: i
        for i, rank in enumerate(
            canonical_ranks_key
        )
    }

    rank_arange = np.arange(
        len(canonical_ranks_key),
        dtype=np.int8,
    )

    canonical_set = frozenset(
        canonical_ranks_key
    )

    return (
        rank_index,
        rank_arange,
        canonical_set,
    )


@lru_cache(maxsize=400_000)
def _cached_taxid_metadata(
    tid: int,
    canonical_ranks_key: tuple,
    canonical_map_obj_id: int,
    ncbi_obj_id: int,
    rank_map_obj_id: int,
):
    """
    Return:
        ancestor_row
        rank_idx

    Normally all information comes from precomputed shared maps,
    so no ETE query occurs in the extraction hot path.
    """

    rank_index, _, _ = _cached_rank_helpers(
        canonical_ranks_key
    )

    canonical_map = (
        globals._shared_canonical_map
        if globals._shared_canonical_map
        is not None
        else {}
    )

    ancs = canonical_map.get(tid)

    if ancs is None:
        ancs = get_canonical_taxid_for_rank(
            tid,
            list(canonical_ranks_key),
            globals.NCBI,
        )

    ancestor_row = tuple(
        (
            int(ancs[rank])
            if ancs.get(rank) is not None
            else -1
        )
        for rank in canonical_ranks_key
    )

    # Prefer rank precomputed by the driver.
    rank_idx_map = getattr(
        globals,
        "_shared_rank_idx_map",
        None,
    )

    if (
        rank_idx_map is not None
        and tid in rank_idx_map
    ):
        rank_idx = int(
            rank_idx_map[tid]
        )

    else:
        raw_rank = get_taxid_rank_raw(
            tid
        )

        canonical_rank = canonicalize_rank(
            raw_rank
        )

        rank_idx = rank_index.get(
            canonical_rank,
            -1,
        )

    return ancestor_row, rank_idx


@dataclass(slots=True)
class TaxonomyLookup:
    canonical_ranks: tuple
    taxids: np.ndarray
    ancestor_table: np.ndarray
    rank_idx: np.ndarray


def _to_unique_taxids(
    taxids: Iterable[int],
) -> np.ndarray:

    if isinstance(taxids, np.ndarray):

        arr = np.asarray(
            taxids,
            dtype=np.int64,
        ).ravel()

    else:

        arr = np.fromiter(
            (int(t) for t in taxids),
            dtype=np.int64,
        )

    if arr.size == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    return np.unique(arr)


def build_taxonomy_lookup(
    taxids: Iterable[int],
    canonical_ranks: Sequence[str],
) -> TaxonomyLookup:

    canonical_ranks_key = tuple(
        canonical_ranks
    )

    unique_taxids = _to_unique_taxids(
        taxids
    )

    n_taxa = len(
        unique_taxids
    )

    n_ranks = len(
        canonical_ranks_key
    )

    ancestor_table = np.full(
        (n_taxa, n_ranks),
        -1,
        dtype=np.int64,
    )

    rank_idx = np.full(
        n_taxa,
        -1,
        dtype=np.int8,
    )

    canonical_map_obj_id = id(
        globals._shared_canonical_map
    )

    ncbi_obj_id = id(
        globals.NCBI
    )

    shared_rank_map = getattr(
        globals,
        "_shared_rank_idx_map",
        None,
    )

    rank_map_obj_id = id(
        shared_rank_map
    )

    for i, tid in enumerate(
        unique_taxids
    ):

        ancestor_row, tax_rank_idx = (
            _cached_taxid_metadata(
                int(tid),
                canonical_ranks_key,
                canonical_map_obj_id,
                ncbi_obj_id,
                rank_map_obj_id,
            )
        )

        ancestor_table[i] = (
            ancestor_row
        )

        rank_idx[i] = (
            tax_rank_idx
        )

    return TaxonomyLookup(
        canonical_ranks=canonical_ranks_key,
        taxids=unique_taxids,
        ancestor_table=ancestor_table,
        rank_idx=rank_idx,
    )


# ============================================================
# Per-worker lookup
# ============================================================

_WORKER_TAXONOMY_LOOKUP = None
_WORKER_TAXONOMY_LOOKUP_KEY = None


def get_worker_taxonomy_lookup(
    canonical_ranks,
):
    """
    Build the compact taxonomy lookup once per worker process.
    """

    global _WORKER_TAXONOMY_LOOKUP
    global _WORKER_TAXONOMY_LOOKUP_KEY

    canonical_map = (
        globals._shared_canonical_map
        or {}
    )

    rank_map = getattr(
        globals,
        "_shared_rank_idx_map",
        None,
    )

    key = (
        id(canonical_map),
        len(canonical_map),
        id(rank_map),
        len(rank_map) if rank_map else 0,
        tuple(canonical_ranks),
    )

    if (
        _WORKER_TAXONOMY_LOOKUP is None
        or
        _WORKER_TAXONOMY_LOOKUP_KEY != key
    ):
        _WORKER_TAXONOMY_LOOKUP = (
            build_taxonomy_lookup(
                canonical_map.keys(),
                canonical_ranks,
            )
        )

        _WORKER_TAXONOMY_LOOKUP_KEY = key

        # Lookup now owns compact numerical arrays.
        # Do not also retain a massive Python LRU representation.
        _cached_taxid_metadata.cache_clear()

    return _WORKER_TAXONOMY_LOOKUP


# ============================================================
# Lookup helpers
# ============================================================

def _find_taxid_indices(
    query_taxids: np.ndarray,
    sorted_taxids: np.ndarray,
):
    if sorted_taxids.size == 0:
        return (
            np.zeros(
                len(query_taxids),
                dtype=np.int64,
            ),
            np.zeros(
                len(query_taxids),
                dtype=bool,
            ),
        )

    idx = np.searchsorted(
        sorted_taxids,
        query_taxids,
    )

    safe_idx = np.minimum(
        idx,
        len(sorted_taxids) - 1,
    )

    valid = (
        (idx < len(sorted_taxids))
        &
        (
            sorted_taxids[safe_idx]
            ==
            query_taxids
        )
    )

    return idx, valid


def _get_taxonomy_arrays(
    taxids,
    canonical_ranks_key,
    taxonomy_lookup=None,
):
    n_taxa = len(taxids)
    n_ranks = len(
        canonical_ranks_key
    )

    ancestors = np.full(
        (n_taxa, n_ranks),
        -1,
        dtype=np.int64,
    )

    rank_idx = np.full(
        n_taxa,
        -1,
        dtype=np.int8,
    )

    found = np.zeros(
        n_taxa,
        dtype=bool,
    )

    if taxonomy_lookup is not None:

        if (
            taxonomy_lookup.canonical_ranks
            != canonical_ranks_key
        ):
            raise ValueError(
                "taxonomy rank mismatch"
            )

        idx, valid = _find_taxid_indices(
            taxids,
            taxonomy_lookup.taxids,
        )

        if np.any(valid):

            ancestors[valid] = (
                taxonomy_lookup
                .ancestor_table[
                    idx[valid]
                ]
            )

            rank_idx[valid] = (
                taxonomy_lookup
                .rank_idx[
                    idx[valid]
                ]
            )

            found[valid] = True

    # Fallback only for unexpected taxids.
    missing = np.flatnonzero(
        ~found
    )

    if missing.size:

        canonical_map_obj_id = id(
            globals._shared_canonical_map
        )

        ncbi_obj_id = id(
            globals.NCBI
        )

        shared_rank_map = getattr(
            globals,
            "_shared_rank_idx_map",
            None,
        )

        rank_map_obj_id = id(
            shared_rank_map
        )

        for i in missing:

            ancestor_row, own_rank = (
                _cached_taxid_metadata(
                    int(taxids[i]),
                    canonical_ranks_key,
                    canonical_map_obj_id,
                    ncbi_obj_id,
                    rank_map_obj_id,
                )
            )

            ancestors[i] = (
                ancestor_row
            )

            rank_idx[i] = (
                own_rank
            )

    return ancestors, rank_idx


# ============================================================
# Candidate context
# ============================================================

@dataclass(slots=True)
class SequenceFeatureContext:
    canonical_ranks: tuple
    taxids: np.ndarray

    # Stored as float32 specifically so NumPy matrix
    # multiplication can use the fast numeric path.
    support_matrix: np.ndarray
    in_matrix: np.ndarray

    pred_anc_arr: np.ndarray
    leaf: Optional[int]


def _build_predicted_ancestor_array(
    pred_lineage,
    canonical_ranks_key,
    lineage_at_rank=None,
):

    _, _, canonical_set = (
        _cached_rank_helpers(
            canonical_ranks_key
        )
    )

    if lineage_at_rank is None:

        lineage_ranks = (
            globals.NCBI.get_rank(
                pred_lineage
            )
        )

        lineage_at_rank = {
            rank: None
            for rank
            in canonical_ranks_key
        }

        for t in pred_lineage:

            raw = lineage_ranks.get(t)

            canonical = canonicalize_rank(
                raw
            )

            if (
                canonical
                in canonical_set
                and
                lineage_at_rank[
                    canonical
                ]
                is None
            ):
                lineage_at_rank[
                    canonical
                ] = int(t)

    arr = np.full(
        len(canonical_ranks_key),
        -1,
        dtype=np.int64,
    )

    for i, rank in enumerate(
        canonical_ranks_key
    ):

        value = lineage_at_rank.get(
            rank
        )

        if value is not None:
            arr[i] = int(value)

    return arr


def prepare_sequence_feature_context(
    sequence_taxids,
    pred_lineage,
    canonical_ranks,
    lineage_at_rank=None,
    taxonomy_lookup=None,
):

    canonical_ranks_key = tuple(
        canonical_ranks
    )

    _, rank_arange, _ = (
        _cached_rank_helpers(
            canonical_ranks_key
        )
    )

    taxids = _to_unique_taxids(
        sequence_taxids
    )

    pred_anc_arr = (
        _build_predicted_ancestor_array(
            pred_lineage,
            canonical_ranks_key,
            lineage_at_rank,
        )
    )

    n_taxa = len(taxids)
    n_ranks = len(
        canonical_ranks_key
    )

    if n_taxa == 0:

        support_matrix = np.empty(
            (0, n_ranks),
            dtype=np.float32,
        )

        in_matrix = np.empty(
            (0, n_ranks),
            dtype=np.float32,
        )

    else:

        ancestor_table, rank_idx = (
            _get_taxonomy_arrays(
                taxids,
                canonical_ranks_key,
                taxonomy_lookup,
            )
        )

        pred_present = (
            pred_anc_arr != -1
        )

        anc_match = (
            ancestor_table
            ==
            pred_anc_arr[None, :]
        )

        support_mask = (
            pred_present[None, :]
            &
            anc_match
        )

        at_node = (
            taxids[:, None]
            ==
            pred_anc_arr[None, :]
        )

        at_rank = (
            rank_idx[:, None]
            ==
            rank_arange[None, :]
        )

        in_mask = (
            support_mask
            &
            (
                at_node
                |
                at_rank
            )
        )

        support_matrix = (
            support_mask.astype(
                np.float32,
                copy=False,
            )
        )

        in_matrix = (
            in_mask.astype(
                np.float32,
                copy=False,
            )
        )

    leaf = (
        int(pred_lineage[-1])
        if pred_lineage
        else None
    )

    return SequenceFeatureContext(
        canonical_ranks=canonical_ranks_key,
        taxids=taxids,
        support_matrix=support_matrix,
        in_matrix=in_matrix,
        pred_anc_arr=pred_anc_arr,
        leaf=leaf,
    )


# ============================================================
# Encode bins ONCE per sequence
# ============================================================

@dataclass(slots=True)
class EncodedSequenceBins:
    taxids: np.ndarray

    # shape:
    #   bins × taxids
    counts: np.ndarray

    # shape:
    #   bins
    totals: np.ndarray


def encode_sequence_bins(
    bins,
    sequence_taxids=None,
):

    if sequence_taxids is None:

        taxid_set = set()

        for bin_counts in bins:
            taxid_set.update(
                int(t)
                for t in bin_counts.keys()
            )

        taxids = np.fromiter(
            taxid_set,
            dtype=np.int64,
            count=len(taxid_set),
        )

        taxids.sort()

    else:

        taxids = _to_unique_taxids(
            sequence_taxids
        )

    n_bins = len(bins)
    n_taxa = len(taxids)

    X = np.zeros(
        (n_bins, n_taxa),
        dtype=np.float32,
    )

    totals = np.zeros(
        n_bins,
        dtype=np.float32,
    )

    if n_taxa == 0:
        return EncodedSequenceBins(
            taxids=taxids,
            counts=X,
            totals=totals,
        )

    for b, bin_counts in enumerate(
        bins
    ):

        if not bin_counts:
            continue

        n = len(bin_counts)

        tids = np.fromiter(
            bin_counts.keys(),
            dtype=np.int64,
            count=n,
        )

        vals = np.fromiter(
            bin_counts.values(),
            dtype=np.float32,
            count=n,
        )

        idx = np.searchsorted(
            taxids,
            tids,
        )

        if (
            np.any(idx >= n_taxa)
            or
            np.any(
                taxids[
                    np.minimum(
                        idx,
                        n_taxa - 1,
                    )
                ]
                != tids
            )
        ):
            raise ValueError(
                "Bin contains taxid absent "
                "from sequence_taxids"
            )

        X[b, idx] = vals

        totals[b] = vals.sum(
            dtype=np.float32
        )

    return EncodedSequenceBins(
        taxids=taxids,
        counts=X,
        totals=totals,
    )


# ============================================================
# All-bin matrix implementation
# ============================================================

def compute_encoded_sequence_features(
    encoded_bins: EncodedSequenceBins,
    feature_context: SequenceFeatureContext,
):

    if not np.array_equal(
        encoded_bins.taxids,
        feature_context.taxids,
    ):
        raise ValueError(
            "Encoded bins and feature context "
            "use different taxid ordering"
        )

    X = encoded_bins.counts
    totals = encoded_bins.totals

    n_bins = X.shape[0]

    n_ranks = len(
        feature_context.canonical_ranks
    )

    result = np.zeros(
        (
            n_bins,
            1 + n_ranks * 3,
        ),
        dtype=np.float32,
    )

    if n_bins == 0:
        return result

    # --------------------------------------------------------
    # The expensive core is now only TWO matrix multiplies
    # for the entire sequence.
    #
    # (bins × taxa) @ (taxa × ranks)
    #          -> (bins × ranks)
    # --------------------------------------------------------

    support = (
        X
        @
        feature_context.support_matrix
    )

    in_lin = (
        X
        @
        feature_context.in_matrix
    )

    desc = (
        support
        -
        in_lin
    )

    out = (
        totals[:, None]
        -
        support
    )

    valid = (
        totals > 0
    )

    denom = np.where(
        valid,
        totals,
        1.0,
    )[:, None]

    result[:, 1::3] = (
        in_lin / denom
    )

    result[:, 2::3] = (
        out / denom
    )

    result[:, 3::3] = (
        desc / denom
    )

    # Leaf feature.
    leaf = feature_context.leaf

    if leaf is not None:

        pos = np.searchsorted(
            encoded_bins.taxids,
            leaf,
        )

        if (
            pos < len(
                encoded_bins.taxids
            )
            and
            encoded_bins.taxids[pos]
            == leaf
        ):
            result[valid, 0] = (
                X[valid, pos]
                /
                totals[valid]
            )

    return result


# ============================================================
# Public sequence API
# ============================================================

def compute_sequence_bin_features(
    bins,
    pred_lineage,
    canonical_ranks,
    lineage_at_rank=None,
    taxonomy_lookup=None,
    sequence_taxids=None,
    encoded_bins=None,
    return_numpy=True,
):

    if taxonomy_lookup is None:
        taxonomy_lookup = (
            get_worker_taxonomy_lookup(
                canonical_ranks
            )
        )

    if encoded_bins is None:

        encoded_bins = (
            encode_sequence_bins(
                bins,
                sequence_taxids,
            )
        )

    context = (
        prepare_sequence_feature_context(
            sequence_taxids=(
                encoded_bins.taxids
            ),
            pred_lineage=pred_lineage,
            canonical_ranks=canonical_ranks,
            lineage_at_rank=lineage_at_rank,
            taxonomy_lookup=taxonomy_lookup,
        )
    )

    result = (
        compute_encoded_sequence_features(
            encoded_bins,
            context,
        )
    )

    if return_numpy:
        return result

    return result.tolist()


# ============================================================
# Backward-compatible single-bin API
# ============================================================

def compute_bin_features(
    kmer_tax_counts,
    pred_lineage,
    canonical_ranks,
    lineage_at_rank=None,
    feature_context=None,
    taxonomy_lookup=None,
    return_numpy=False,
):

    if taxonomy_lookup is None:
        taxonomy_lookup = (
            get_worker_taxonomy_lookup(
                canonical_ranks
            )
        )

    if feature_context is None:

        feature_context = (
            prepare_sequence_feature_context(
                sequence_taxids=(
                    kmer_tax_counts.keys()
                ),
                pred_lineage=pred_lineage,
                canonical_ranks=canonical_ranks,
                lineage_at_rank=lineage_at_rank,
                taxonomy_lookup=taxonomy_lookup,
            )
        )

    encoded = encode_sequence_bins(
        [kmer_tax_counts],
        sequence_taxids=(
            feature_context.taxids
        ),
    )

    vec = (
        compute_encoded_sequence_features(
            encoded,
            feature_context,
        )[0]
    )

    if return_numpy:
        return vec

    return vec.tolist()


# ============================================================
# Cache reset
# ============================================================

def clear_feature_caches():

    global _WORKER_TAXONOMY_LOOKUP
    global _WORKER_TAXONOMY_LOOKUP_KEY

    _WORKER_TAXONOMY_LOOKUP = None
    _WORKER_TAXONOMY_LOOKUP_KEY = None

    _cached_rank_helpers.cache_clear()
    _cached_taxid_metadata.cache_clear()
    _resample_plan.cache_clear()
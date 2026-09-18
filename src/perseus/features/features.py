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
    """
    Precompute interpolation indices/weights for vectorized resampling.
    """
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
    """
    Resample the T dimension of an array of shape (T, C)
    to T_target using vectorized linear interpolation.

    Parameters
    ----------
    x_TN
        Input array with shape (T, C).
    T_target
        Target number of positions.

    Returns
    -------
    np.ndarray
        Array with shape (T_target, C).
    """
    T, C = x_TN.shape

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

    out = (
        x[left] * (1.0 - weight)
        + x[right] * weight
    )

    return out.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# Torch helpers
# ============================================================

def _torch_dtype(name: str) -> torch.dtype:
    """
    Convert string name to torch.dtype.
    """
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
# Rank / taxonomy caches
# ============================================================

@lru_cache(maxsize=128)
def _cached_rank_helpers(
    canonical_ranks_key: tuple,
):
    """
    Cache small rank-related objects that otherwise would be
    reconstructed for every bin.
    """
    rank_index = {
        rank: i
        for i, rank in enumerate(canonical_ranks_key)
    }

    rank_arange = np.arange(
        len(canonical_ranks_key),
        dtype=np.int64,
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
):
    """
    Return:

        ancestor_row:
            canonical ancestor taxid at every rank

        rank_idx:
            canonical rank index of this taxid

    The object IDs are included in the cache key so the cache
    does not accidentally reuse taxonomy information if the
    underlying taxonomy objects are replaced.
    """
    rank_index, _, _ = _cached_rank_helpers(
        canonical_ranks_key
    )

    canonical_map = (
        globals._shared_canonical_map
        if globals._shared_canonical_map is not None
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

    raw_rank = get_taxid_rank_raw(tid)

    canonical_rank = canonicalize_rank(
        raw_rank
    )

    rank_idx = rank_index.get(
        canonical_rank,
        -1,
    )

    return ancestor_row, rank_idx


# ============================================================
# Compact process/chunk-level taxonomy lookup
# ============================================================

@dataclass(slots=True)
class TaxonomyLookup:
    """
    Compact numerical representation of taxonomy information.

    taxids
        Sorted taxids.

    ancestor_table
        Shape:
            (n_taxids, n_ranks)

        ancestor_table[i, r] gives the ancestor of taxid i
        at canonical rank r.

    rank_idx
        Shape:
            (n_taxids,)

        Canonical rank index for each taxid.
    """

    canonical_ranks: tuple

    taxids: np.ndarray

    ancestor_table: np.ndarray

    rank_idx: np.ndarray


def _to_unique_taxids(
    taxids: Iterable[int],
) -> np.ndarray:
    """
    Convert iterable of taxids to sorted unique int64 array.
    """
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
    """
    Build a compact taxonomy lookup for all taxids expected
    during feature extraction.

    Ideally call this once per worker/chunk rather than once
    per sequence.
    """
    canonical_ranks_key = tuple(
        canonical_ranks
    )

    unique_taxids = _to_unique_taxids(
        taxids
    )

    n_taxa = len(unique_taxids)
    n_ranks = len(canonical_ranks_key)

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

    for i, tid in enumerate(unique_taxids):

        ancestor_row, tax_rank_idx = (
            _cached_taxid_metadata(
                int(tid),
                canonical_ranks_key,
                canonical_map_obj_id,
                ncbi_obj_id,
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
# Compact lookup helpers
# ============================================================

def _find_taxid_indices(
    query_taxids: np.ndarray,
    sorted_taxids: np.ndarray,
):
    """
    Map taxids to rows of a sorted taxid lookup using
    np.searchsorted.

    Returns
    -------
    idx
        Row indices.

    valid
        Boolean array indicating which taxids were found.
    """
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
            == query_taxids
        )
    )

    return idx, valid


def _get_taxonomy_arrays(
    taxids: np.ndarray,
    canonical_ranks_key: tuple,
    taxonomy_lookup: Optional[TaxonomyLookup] = None,
):
    """
    Get ancestor rows and own-rank indices for an array of taxids.

    Uses the compact lookup when available and falls back to the
    cached taxonomy helper for any missing taxids.
    """
    n_taxa = len(taxids)
    n_ranks = len(canonical_ranks_key)

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

    found = np.zeros(
        n_taxa,
        dtype=bool,
    )

    # ----------------------------------------
    # Fast compact lookup
    # ----------------------------------------

    if taxonomy_lookup is not None:

        if (
            taxonomy_lookup.canonical_ranks
            != canonical_ranks_key
        ):
            raise ValueError(
                "taxonomy_lookup canonical ranks "
                "do not match canonical_ranks"
            )

        idx, valid = _find_taxid_indices(
            taxids,
            taxonomy_lookup.taxids,
        )

        if np.any(valid):
            ancestor_table[valid] = (
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

    # ----------------------------------------
    # Cached fallback
    # ----------------------------------------

    missing_idx = np.flatnonzero(
        ~found
    )

    if missing_idx.size:

        canonical_map_obj_id = id(
            globals._shared_canonical_map
        )

        ncbi_obj_id = id(
            globals.NCBI
        )

        for i in missing_idx:

            tid = int(
                taxids[i]
            )

            ancestor_row, tax_rank_idx = (
                _cached_taxid_metadata(
                    tid,
                    canonical_ranks_key,
                    canonical_map_obj_id,
                    ncbi_obj_id,
                )
            )

            ancestor_table[i] = (
                ancestor_row
            )

            rank_idx[i] = (
                tax_rank_idx
            )

    return (
        ancestor_table,
        rank_idx,
    )


# ============================================================
# Sequence-level context
# ============================================================

@dataclass(slots=True)
class SequenceFeatureContext:
    """
    Taxonomic relationships that are constant for every bin
    belonging to one sequence.

    Instead of recomputing lineage relationships for every
    1-kb bin, calculate them once here.
    """

    canonical_ranks: tuple

    taxids: np.ndarray

    support_matrix: np.ndarray

    in_matrix: np.ndarray

    pred_anc_arr: np.ndarray

    leaf: Optional[int]


def _build_predicted_ancestor_array(
    pred_lineage: Sequence[int],
    canonical_ranks_key: tuple,
    lineage_at_rank=None,
):
    """
    Build the predicted lineage ancestor vector once.
    """
    _, _, canonical_set = (
        _cached_rank_helpers(
            canonical_ranks_key
        )
    )

    if lineage_at_rank is None:

        lineage_ranks = globals.NCBI.get_rank(
            pred_lineage
        )

        lineage_at_rank = {
            rank: None
            for rank in canonical_ranks_key
        }

        for t in pred_lineage:

            raw = lineage_ranks.get(t)

            canonical = canonicalize_rank(
                raw
            )

            if (
                canonical in canonical_set
                and
                lineage_at_rank[canonical]
                is None
            ):
                lineage_at_rank[
                    canonical
                ] = int(t)

    pred_anc_arr = np.full(
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
            pred_anc_arr[i] = int(
                value
            )

    return pred_anc_arr


def prepare_sequence_feature_context(
    sequence_taxids: Iterable[int],
    pred_lineage: Sequence[int],
    canonical_ranks: Sequence[str],
    lineage_at_rank=None,
    taxonomy_lookup: Optional[TaxonomyLookup] = None,
) -> SequenceFeatureContext:
    """
    Precompute all relationships between taxids occurring in a
    sequence and the sequence's predicted lineage.

    This should be called ONCE PER SEQUENCE.

    Then call compute_bin_features(..., feature_context=context)
    for every bin belonging to that sequence.
    """
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
            dtype=bool,
        )

        in_matrix = np.empty(
            (0, n_ranks),
            dtype=bool,
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

        # ------------------------------------
        # Does this taxid belong to the
        # predicted lineage at each rank?
        # ------------------------------------

        anc_match = (
            ancestor_table
            ==
            pred_anc_arr[None, :]
        )

        support_matrix = (
            pred_present[None, :]
            &
            anc_match
        )

        # ------------------------------------
        # Is the kmer taxid itself the
        # predicted node at the rank?
        # ------------------------------------

        at_node = (
            taxids[:, None]
            ==
            pred_anc_arr[None, :]
        )

        # ------------------------------------
        # Is the taxid itself assigned at
        # this canonical rank?
        # ------------------------------------

        at_rank = (
            rank_idx[:, None]
            ==
            rank_arange[None, :]
        )

        in_matrix = (
            support_matrix
            &
            (
                at_node
                |
                at_rank
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
# Bin-level feature extraction
# ============================================================

def _compute_bin_features_fast(
    kmer_tax_counts: Mapping[int, int],
    feature_context: SequenceFeatureContext,
) -> np.ndarray:
    """
    Fast bin-level feature computation.

    Assumes taxonomic relationships were already calculated
    in prepare_sequence_feature_context().
    """
    n_ranks = len(
        feature_context.canonical_ranks
    )

    n_taxa = len(
        kmer_tax_counts
    )

    # ----------------------------------------
    # Empty bin
    # ----------------------------------------

    if n_taxa == 0:
        return np.zeros(
            1 + n_ranks * 3,
            dtype=np.float32,
        )

    # ----------------------------------------
    # Dict -> compact numerical arrays
    # ----------------------------------------

    taxids = np.fromiter(
        (
            int(t)
            for t in kmer_tax_counts.keys()
        ),
        dtype=np.int64,
        count=n_taxa,
    )

    counts = np.fromiter(
        kmer_tax_counts.values(),
        dtype=np.float32,
        count=n_taxa,
    )

    total = float(
        counts.sum()
    )

    if total <= 0:
        return np.zeros(
            1 + n_ranks * 3,
            dtype=np.float32,
        )

    # ----------------------------------------
    # Map taxids onto precomputed sequence
    # relationships.
    # ----------------------------------------

    idx, valid = _find_taxid_indices(
        taxids,
        feature_context.taxids,
    )

    if not np.all(valid):

        missing = taxids[
            ~valid
        ]

        raise ValueError(
            "SequenceFeatureContext does not contain "
            "all taxids in this bin. Missing taxids: "
            f"{missing[:20].tolist()}"
        )

    support_matrix = (
        feature_context
        .support_matrix[
            idx
        ]
    )

    in_matrix = (
        feature_context
        .in_matrix[
            idx
        ]
    )

    # ----------------------------------------
    # Only TWO reductions are needed.
    #
    # support = in + descendants
    #
    # therefore:
    #
    # desc = support - in
    # out  = total - support
    # ----------------------------------------

    support = (
        counts
        @
        support_matrix
    )

    in_lin = (
        counts
        @
        in_matrix
    )

    desc = (
        support
        -
        in_lin
    )

    out = (
        total
        -
        support
    )

    # ----------------------------------------
    # Final feature vector
    # ----------------------------------------

    vec = np.empty(
        1 + n_ranks * 3,
        dtype=np.float32,
    )

    leaf = (
        feature_context.leaf
    )

    vec[0] = (
        kmer_tax_counts.get(
            leaf,
            0,
        )
        /
        total
    )

    vec[1::3] = (
        in_lin
        /
        total
    )

    vec[2::3] = (
        out
        /
        total
    )

    vec[3::3] = (
        desc
        /
        total
    )

    return vec


# ============================================================
# Backward-compatible public function
# ============================================================

def compute_bin_features(
    kmer_tax_counts,
    pred_lineage,
    canonical_ranks,
    lineage_at_rank=None,
    feature_context: Optional[
        SequenceFeatureContext
    ] = None,
    taxonomy_lookup: Optional[
        TaxonomyLookup
    ] = None,
    return_numpy: bool = False,
):
    """
    Compute feature vector for one bin.

    FASTEST USAGE
    -------------
    Prepare a SequenceFeatureContext once per sequence and pass
    it into every bin:

        context = prepare_sequence_feature_context(...)

        feature = compute_bin_features(
            counts,
            pred_lineage,
            canonical_ranks,
            feature_context=context,
            return_numpy=True,
        )

    BACKWARD-COMPATIBLE USAGE
    -------------------------
    Calling this exactly like the previous implementation still
    works, but it cannot reuse relationships across bins.

    Parameters
    ----------
    kmer_tax_counts
        Mapping taxid -> count.

    pred_lineage
        Predicted lineage taxids.

    canonical_ranks
        Ordered canonical ranks.

    lineage_at_rank
        Optional precomputed rank -> taxid mapping.

    feature_context
        Optional sequence-level precomputed context.

    taxonomy_lookup
        Optional process/chunk-level compact taxonomy table.

    return_numpy
        If True, return np.ndarray instead of converting to a
        Python list.

    Returns
    -------
    list or np.ndarray
        Feature vector.
    """

    # ----------------------------------------
    # Backward-compatible slow path.
    #
    # Build context using only the current
    # bin. Correct, but less efficient.
    # ----------------------------------------

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

    vec = _compute_bin_features_fast(
        kmer_tax_counts,
        feature_context,
    )

    if return_numpy:
        return vec

    return vec.tolist()


# ============================================================
# Whole-sequence convenience function
# ============================================================

def compute_sequence_bin_features(
    bins,
    pred_lineage,
    canonical_ranks,
    lineage_at_rank=None,
    taxonomy_lookup=None,
    sequence_taxids=None,
    return_numpy=True,
):
    """
    Compute feature vectors for every bin in one sequence.

    Taxonomic metadata can be precomputed once for the sequence
    and reused across multiple candidate predicted taxa.
    """

    n_bins = len(bins)

    n_features = (
        1
        +
        len(canonical_ranks) * 3
    )

    if n_bins == 0:
        result = np.empty(
            (0, n_features),
            dtype=np.float32,
        )

        if return_numpy:
            return result

        return result.tolist()

    # Only collect taxids here if caller did not already do so.
    if sequence_taxids is None:
        taxid_set = set()

        for bin_counts in bins:
            taxid_set.update(
                int(t)
                for t in bin_counts.keys()
            )

        sequence_taxids = np.fromiter(
            taxid_set,
            dtype=np.int64,
            count=len(taxid_set),
        )

    context = prepare_sequence_feature_context(
        sequence_taxids=sequence_taxids,
        pred_lineage=pred_lineage,
        canonical_ranks=canonical_ranks,
        lineage_at_rank=lineage_at_rank,
        taxonomy_lookup=taxonomy_lookup,
    )

    output = np.empty(
        (
            n_bins,
            n_features,
        ),
        dtype=np.float32,
    )

    for i, bin_counts in enumerate(bins):
        output[i] = _compute_bin_features_fast(
            bin_counts,
            context,
        )

    if return_numpy:
        return output

    return output.tolist()


# ============================================================
# Cache management
# ============================================================

def clear_feature_caches():
    """
    Clear feature-extraction caches.

    Useful if globals.NCBI or the shared taxonomy map is
    replaced during the lifetime of a process.
    """
    _cached_rank_helpers.cache_clear()
    _cached_taxid_metadata.cache_clear()
    _resample_plan.cache_clear()
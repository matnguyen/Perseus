import re
import pandas as pd
import numpy as np
import logging
import random
import multiprocessing as mp

from functools import lru_cache
from alive_progress import alive_bar
from collections import defaultdict

import perseus.utils.globals as globals

from perseus.features.features import (
    compute_bin_features,
    compute_sequence_bin_features,
    encode_sequence_bins,
    get_worker_taxonomy_lookup,
)

from perseus.utils.constants import (
    CANONICAL_RANKS,
)

from perseus.utils.tax_utils import (
    get_ncbi,
    normalize_taxid,
    get_lineage_path,
    lineage_to_rank_map,
)

from perseus.features.init import (
    _init_ncbi_private_db,
    effective_nprocs,
)

from perseus.utils.io_utils import (
    _write_rows_streaming_shards,
    prefetch,
)

logger = logging.getLogger(__name__)


def parse_kmers(s):
    """
    Parse Kraken2 kmer classification into list of "taxid:count" strings

    Args:
        s (str): Kraken2 kmer string

    Returns:
        list: List of "taxid:count" strings
    """
    if pd.isna(s):
        return []
    return [x for x in s.split() if ':' in x]


def extract_tax_counts(kmer_list):
    """
    Extract kmer counts from list of "taxid:count" strings

    Args:
        kmer_list (list): List of "taxid:count" strings

    Returns:
        dict: Mapping of taxid to total kmer count
    """
    tax_counts = defaultdict(int)
    for kmer in kmer_list:
        try:
            tax_str, count_str = kmer.split(':', 1)
            tax_norm = normalize_taxid(int(tax_str))
            if tax_norm is None:
                continue
            count = int(count_str)
        except Exception:
            # Any parsing error: skip
            continue
        
        # Ignore non-positive counts
        if count <= 0:
            continue
        
        tax_counts[tax_norm] += count
    return tax_counts


def extract_tax_context_chunk(chunk):
    """
    Extract kmer taxonomic counts for each sequence in a chunk

    Args:
        chunk (pd.DataFrame): DataFrame chunk with at least columns 'ID' and 'Kmers'

    Returns:
        dict: Mapping from sequence ID to dict of taxid → kmer count
    """
    results = {}
    logger.debug("Processing chunk in extract_tax_context_chunk.")
    chunk = chunk.dropna(subset=['Kmers'])
    for row in chunk.itertuples(index=False):
        kmers = parse_kmers(row.Kmers)
        results[row.ID] = extract_tax_counts(kmers)
    return results


# Parse "taxid:count" tokens lazily (no full split/explode)
TOKEN_RE = re.compile(r'(\d+):(\d+)')

def iter_kmer_tokens(kmers_str):
    """
    Yield (taxid:int, count:int) one-by-one from the Kmers string.

    Args:
        kmers_str (str): Kraken2 kmer string

    Yields:
        tuple[int, int]: (taxid, count) pairs extracted from the string
    """
    if not isinstance(kmers_str, str) or not kmers_str:
        return
    for m in TOKEN_RE.finditer(kmers_str):
        yield int(m.group(1)), int(m.group(2))


def add_to_bins(bin_counts_by_bin, bin_size, taxid, count, cur_pos):
    """
    Distribute `count` for `taxid` across fixed-size bins starting at cur_pos

    Mutates bin_counts_by_bin in-place: {bin_idx -> {taxid -> kmer_count}}

    Args:
        bin_counts_by_bin (dict): Dictionary mapping bin index to {taxid: kmer_count}
        bin_size (int): Size of each bin
        taxid (int): Taxonomic ID to distribute
        count (int): Number of k-mers to distribute
        cur_pos (int): Current position (start index for distribution)

    Returns:
        int: The new position after distributing all counts
    """
    pos = cur_pos
    remaining = int(count)
    while remaining > 0:
        bin_idx = pos // bin_size
        offset  = pos %  bin_size
        space   = bin_size - offset
        take    = remaining if remaining <= space else space
        d = bin_counts_by_bin.get(bin_idx)
        if d is None:
            d = {}
            bin_counts_by_bin[bin_idx] = d
        d[taxid] = d.get(taxid, 0) + take
        pos      += take
        remaining -= take
    return pos


def build_tax_context(file_path, db_path, rows_per_chunk=1000, prefetch_buf=64, dispatch_batch=4, threads=0):
    """
    Build a mapping from sequence ID to k-mer taxonomic counts from a Kraken2 classification file

    Args:
        file_path (str): Path to the Kraken2 classification file (TSV format)
        rows_per_chunk (int, optional): Number of rows to process per chunk. Defaults to 1000
        prefetch_buf (int, optional): Buffer size for prefetching chunks. Defaults to 64
        dispatch_batch (int, optional): Number of chunks to dispatch per worker batch. Defaults to 4
        threads (int, optional): Number of worker processes to use (0 = auto). Defaults to 0

    Returns:
        dict: Mapping from sequence ID (str) to dict of taxid (int) to k-mer count (int)
    """
    logger.info("Precomputing taxonomic context from full Kraken file")
    tax_context = {}

    reader = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["Classified", "ID", "Taxonomy", "Length", "Kmers"],
        usecols=["ID", "Taxonomy", "Kmers"],
        dtype={"ID": "string", "Taxonomy": "string", "Kmers": "string"},
        chunksize=rows_per_chunk,
        engine="c"
    )

    # Decide number of workers based on `threads` param (0 = auto)
    if threads == 0:
        nprocs = effective_nprocs()
    else:
        nprocs = int(threads)

    if nprocs <= 1:
        logger.info("Building tax_context in single-threaded mode.")
        # Ensure NCBI handle is initialized in this process
        globals.NCBI = get_ncbi(db_path)
        with alive_bar(title="Building tax context", unknown="dots_waves") as bar:
            for chunk in reader:
                res = extract_tax_context_chunk(chunk)
                if res:
                    tax_context.update(res)
                bar()
        return tax_context

    logger.info(f"Using {nprocs} processes for tax_context build.")
    with mp.Pool(processes=nprocs, maxtasksperchild=200, initializer=_init_ncbi_private_db, initargs=(db_path,)) as pool:
        it = pool.imap_unordered(
            extract_tax_context_chunk,
            prefetch(reader, bufsize=prefetch_buf),
            chunksize=dispatch_batch
        )
        pending = []
        with alive_bar(title="Building tax context", unknown="dots_waves") as bar:
            for result in it:
                pending.append(result)
                if len(pending) >= 32:
                    for d in pending:
                        tax_context.update(d)
                    pending.clear()
                bar()
        for d in pending:
            tax_context.update(d)

    return tax_context

@lru_cache(maxsize=8)
def _load_mess_taxid_map(
    mess_true_file,
    mess_input_file,
):
    """
    Read MESS mappings once per worker process.

    Returns:
        dict[str, taxid]
    """

    if not mess_true_file:
        return {}

    if (
        mess_input_file
        and
        mess_true_file == mess_input_file
    ):

        mess_map = pd.read_csv(
            mess_true_file,
            sep="\t",
            header=None,
            names=[
                "seq_id",
                "ref",
            ],
        )

        mess_map["tax_id"] = (
            mess_map["ref"]
            .str.split("|")
            .str[1]
        )

    elif mess_input_file:

        mess_df = pd.read_csv(
            mess_true_file,
            sep="\t",
            header=None,
            names=[
                "seq_id",
                "name",
            ],
        )

        mess_df["name"] = (
            mess_df["name"]
            .str.split("/")
            .str[0]
            .str[:-1]
        )

        mess_df["name"] = (
            mess_df["name"]
            .str.replace(
                r"\.(\d)\d*(?:_.*)?",
                r".\1",
                regex=True,
            )
        )

        mess_input_df = pd.read_csv(
            mess_input_file,
            sep="\t",
            header=0,
        )

        if (
            mess_input_df["tax_id"] == 0
        ).all():

            mess_input_df["tax_id"] = (
                mess_input_df["fasta"]
                .str.split("__")
                .str[2]
            )

        mess_input_df["fasta"] = (
            mess_input_df["fasta"]
            .str.split("__")
            .str[-1]
        )

        mess_map = (
            mess_df
            .set_index("name")
            .join(
                mess_input_df
                .set_index("fasta")[
                    ["tax_id"]
                ],
                how="left",
            )
            .reset_index()
        )

    else:
        # Generic two-column true mapping fallback.
        mess_map = pd.read_csv(
            mess_true_file,
            sep="\t",
            header=None,
        )

        if mess_map.shape[1] < 2:
            return {}

        mess_map = mess_map.iloc[:, :2]
        mess_map.columns = [
            "seq_id",
            "tax_id",
        ]

    if "seq_id" not in mess_map:
        return {}

    return {
        str(seq_id): taxid
        for seq_id, taxid in zip(
            mess_map["seq_id"],
            mess_map["tax_id"],
        )
    }


@lru_cache(maxsize=200_000)
def _cached_true_rank_tuple(
    true_tax: int,
):
    """
    Cache true lineage/rank mapping because many simulated
    sequences typically share the same true taxon.
    """

    lineage = (
        globals._shared_lineage_map.get(
            int(true_tax),
            (),
        )
    )

    if not lineage:
        lineage = get_lineage_path(
            int(true_tax)
        )

    if not lineage:
        return None

    rank_map = lineage_to_rank_map(
        lineage,
        CANONICAL_RANKS,
    )

    return tuple(
        rank_map.get(rank)
        for rank in CANONICAL_RANKS
    )


def process_chunk_iter(
    chunk,
    bin_size=1000,
    topk_taxa=8,
    min_tax_kmers=10,
    max_bins_per_seq=None,
    mess_true_file=None,
    mess_input_file=None,
    neg_extra=4,
    keep_taxonomy=True,
    seed=0,
    is_training=False,
):

    if chunk.empty:
        return

    view = chunk.loc[
        chunk["Classified"] == "C",
        [
            "ID",
            "Length",
            "Kmers",
            "Taxonomy",
        ],
    ]

    if view.empty:
        return

    rng = random.Random(seed)

    # Built only once on the first chunk handled
    # by this worker.
    taxonomy_lookup = (
        get_worker_taxonomy_lookup(
            CANONICAL_RANKS
        )
    )

    if mess_true_file:
        try:
            mess_taxid_map = (
                _load_mess_taxid_map(
                    mess_true_file,
                    mess_input_file,
                )
            )
        except Exception:
            logger.exception(
                "Could not load MESS mapping"
            )
            mess_taxid_map = {}
    else:
        mess_taxid_map = {}

    shared_canonical = (
        globals._shared_canonical_map
        or {}
    )

    for row in view.itertuples(
        index=False
    ):

        seq_id = row.ID
        kmers_str = row.Kmers

        if (
            not isinstance(
                kmers_str,
                str,
            )
            or
            not kmers_str
        ):
            continue

        # ====================================================
        # True taxon
        # ====================================================

        if mess_true_file:

            true_tax_raw = (
                mess_taxid_map.get(
                    str(seq_id)
                )
            )

            if true_tax_raw is None:

                try:
                    true_tax_raw = (
                        row.ID.split("|")[1]
                    )
                except Exception:
                    true_tax_raw = (
                        row.Taxonomy
                    )

        else:

            try:
                true_tax_raw = (
                    row.ID.split("|")[1]
                )
            except Exception:
                true_tax_raw = (
                    row.Taxonomy
                )

        try:
            true_tax = normalize_taxid(
                true_tax_raw
            )
        except Exception:
            continue

        if true_tax is None:
            continue

        true_tax = int(
            true_tax
        )

        true_rank_tuple = (
            _cached_true_rank_tuple(
                true_tax
            )
        )

        if true_rank_tuple is None:
            continue

        # ====================================================
        # Parse Kraken k-mer evidence once
        # ====================================================

        bin_counts_by_bin = {}

        tax_totals = defaultdict(
            int
        )

        cur_pos = 0

        for taxid, count in iter_kmer_tokens(
            kmers_str
        ):

            tax_totals[taxid] += count

            cur_pos = add_to_bins(
                bin_counts_by_bin,
                bin_size,
                taxid,
                count,
                cur_pos,
            )

        if not bin_counts_by_bin:
            continue

        # ====================================================
        # Candidate taxa
        # ====================================================

        candidates = [
            t
            for t, c
            in tax_totals.items()
            if c >= min_tax_kmers
        ]

        if not candidates:
            continue

        candidates.sort(
            key=tax_totals.get,
            reverse=True,
        )

        keep_set = set()

        if keep_taxonomy:

            try:
                keep = normalize_taxid(
                    row.Taxonomy
                )

                if keep is not None:
                    keep_set.add(
                        int(keep)
                    )

            except Exception:
                pass

        if is_training:

            if (
                topk_taxa is not None
                and
                topk_taxa > 0
                and
                len(candidates)
                > topk_taxa
            ):

                top = candidates[
                    :topk_taxa
                ]

                tail = candidates[
                    topk_taxa:
                ]

            else:

                top = candidates
                tail = []

            if neg_extra and tail:

                m = min(
                    int(neg_extra),
                    len(tail),
                )

                extra = rng.sample(
                    tail,
                    m,
                )

            else:

                extra = []

            selected_candidates = []
            seen = set()

            for t in (
                top + extra
            ):

                if t not in seen:
                    seen.add(t)

                    selected_candidates.append(
                        t
                    )

            for t in keep_set:

                if (
                    t not in seen
                    and
                    t in tax_totals
                ):

                    seen.add(t)

                    selected_candidates.append(
                        t
                    )

            candidates = (
                selected_candidates
            )

        # ====================================================
        # Bin downsampling
        # ====================================================

        bin_indices = sorted(
            bin_counts_by_bin
        )

        if (
            max_bins_per_seq
            and
            len(bin_indices)
            >
            max_bins_per_seq
        ):

            step = (
                len(bin_indices)
                /
                max_bins_per_seq
            )

            bin_indices = [
                bin_indices[
                    int(i * step)
                ]
                for i
                in range(
                    max_bins_per_seq
                )
            ]

        bins_for_sequence = [
            bin_counts_by_bin[b]
            for b in bin_indices
        ]

        # ====================================================
        # Encode bins exactly ONCE for this sequence
        # ====================================================

        sequence_taxid_set = set()

        for bin_counts in (
            bins_for_sequence
        ):
            sequence_taxid_set.update(
                bin_counts.keys()
            )

        sequence_taxids = np.fromiter(
            sequence_taxid_set,
            dtype=np.int64,
            count=len(
                sequence_taxid_set
            ),
        )

        sequence_taxids.sort()

        encoded_bins = (
            encode_sequence_bins(
                bins_for_sequence,
                sequence_taxids=(
                    sequence_taxids
                ),
            )
        )

        # ====================================================
        # Candidate-specific calculations
        # ====================================================

        for pred_tax in candidates:

            pred_tax = normalize_taxid(
                pred_tax
            )

            if pred_tax is None:
                continue

            pred_tax = int(
                pred_tax
            )

            pred_lineage = (
                globals
                ._shared_lineage_map
                .get(
                    pred_tax,
                    (),
                )
            )

            if not pred_lineage:
                continue

            # Prefer already-precomputed canonical map.
            pred_at_rank = (
                shared_canonical.get(
                    pred_tax
                )
            )

            if pred_at_rank is None:

                pred_at_rank = (
                    lineage_to_rank_map(
                        pred_lineage,
                        CANONICAL_RANKS,
                    )
                )

            labels_per_rank = []

            for ri, rank in enumerate(
                CANONICAL_RANKS
            ):

                tr = (
                    true_rank_tuple[ri]
                )

                pr = (
                    pred_at_rank.get(
                        rank
                    )
                )

                labels_per_rank.append(
                    int(
                        tr is not None
                        and
                        pr is not None
                        and
                        tr == pr
                    )
                )

            # No bin dictionary parsing here.
            # No per-bin searchsorted here.
            # Two matrix multiplies handle all bins.
            bins_vecs = (
                compute_sequence_bin_features(
                    bins=bins_for_sequence,
                    pred_lineage=pred_lineage,
                    canonical_ranks=(
                        CANONICAL_RANKS
                    ),
                    lineage_at_rank=(
                        pred_at_rank
                    ),
                    taxonomy_lookup=(
                        taxonomy_lookup
                    ),
                    sequence_taxids=(
                        sequence_taxids
                    ),
                    encoded_bins=(
                        encoded_bins
                    ),
                    return_numpy=False,
                )
            )

            yield {
                "seq_id": seq_id,
                "taxon": pred_tax,
                "true_taxon": true_tax,
                "bins": bins_vecs,
                "labels_per_rank": (
                    labels_per_rank
                ),
            }



def process_chunk_and_write(
        chunk, 
        max_bins_per_seq=None,
        mess_true_file=None, 
        mess_input_file=None,
        topk_taxa=8, 
        min_tax_kmers=10, 
        neg_extra=4,
        is_training=False
    ):
    """
    Process a chunk of classified sequences, bin features, generate labels, and write output shards or parquet files

    Args:
        chunk (pd.DataFrame): DataFrame with at least columns 'ID', 'Length', 'Kmers', 'Taxonomy'
        max_bins_per_seq (int or None, optional): Maximum number of bins per sequence (downsample if exceeded). Defaults to None
        mess_true_file (str or None, optional): Path to MESS true mapping file for ground truth taxids. Defaults to None
        mess_input_file (str or None, optional): Path to MESS input mapping file for predicted taxids. Defaults to None

    Returns:
        dict or None: Metadata dictionary from the last written batch, or None if nothing was written
    """
    logger.debug(f"Processing chunk in process_chunk_and_write.")
    for_rows = []
    total_rows = 0
    wrote_meta = None
    batch_idx = 0

    # Controls copied from shared globals
    shard_size = globals._shared_shard_size
    target_len = globals._shared_target_length
    to_dtype   = globals._shared_to_dtype
    
    if is_training:
        logger.debug("Processing in training mode.")
        for row in process_chunk_iter(
                chunk, bin_size=1000, 
                topk_taxa=topk_taxa, 
                min_tax_kmers=min_tax_kmers, 
                neg_extra=neg_extra, 
                max_bins_per_seq=max_bins_per_seq,
                mess_true_file=mess_true_file, 
                mess_input_file=mess_input_file,
                is_training=is_training
            ):
            for_rows.append(row)
            need_flush = False
            need_flush = (len(for_rows) >= shard_size)

            if need_flush:
                logger.debug(f"Writing batch {batch_idx} with {len(for_rows)} rows.")
                meta = _write_rows_streaming_shards(for_rows, max_batch_rows=shard_size,
                                                    target_length=target_len, to_dtype=to_dtype)
                logger.debug(f"Batch {batch_idx} written: {meta}")
                total_rows += meta.get('rows', 0) if meta else 0
                wrote_meta = meta
                for_rows.clear()
                batch_idx += 1

        if for_rows:
            logger.debug(f"Writing final batch {batch_idx} with {len(for_rows)} rows.")
            meta_tail = _write_rows_streaming_shards(for_rows, max_batch_rows=shard_size,
                                                     target_length=target_len, to_dtype=to_dtype)
            logger.debug(f"Final batch written: {meta_tail}")
            total_rows += meta_tail.get('rows', 0) if meta_tail else 0
            wrote_meta = meta_tail or wrote_meta

        logger.debug(f"Total rows written in process_chunk_and_write: {total_rows}")
        return wrote_meta
    else:
        logger.debug("Processing in regular mode.")
        for row in process_chunk_iter(
                chunk, 
                bin_size=1000, 
                min_tax_kmers=min_tax_kmers, 
                max_bins_per_seq=max_bins_per_seq,
                is_training=is_training
            ):
            for_rows.append(row)
            need_flush = False
            need_flush = (len(for_rows) >= shard_size)

            if need_flush:
                logger.debug(f"Writing batch {batch_idx} with {len(for_rows)} rows.")
                meta = _write_rows_streaming_shards(for_rows, max_batch_rows=shard_size,
                                                    target_length=target_len, to_dtype=to_dtype)
                logger.debug(f"Batch {batch_idx} written: {meta}")
                total_rows += meta.get('rows', 0) if meta else 0
                wrote_meta = meta
                for_rows.clear()
                batch_idx += 1

        if for_rows:
            logger.debug(f"Writing final batch {batch_idx} with {len(for_rows)} rows.")
            meta_tail = _write_rows_streaming_shards(for_rows, max_batch_rows=shard_size,
                                                     target_length=target_len, to_dtype=to_dtype)
            logger.debug(f"Final batch written: {meta_tail}")
            total_rows += meta_tail.get('rows', 0) if meta_tail else 0
            wrote_meta = meta_tail or wrote_meta

        logger.debug(f"Total rows written in process_chunk_and_write: {total_rows}")
        return wrote_meta


def process_chunk_and_write_wrapper(args):
    """
    Wrapper to call process_chunk_and_write with unpacked arguments

    Args:
        args (tuple): Tuple containing:
            chunk (pd.DataFrame): DataFrame with at least columns 'ID', 'Length', 'Kmers', 'Taxonomy'
            max_bins_per_seq (int or None): Maximum number of bins per sequence (downsample if exceeded)
            mess_true_file (str or None): Path to MESS true mapping file for ground truth taxids
            mess_input_file (str or None): Path to MESS input mapping file for predicted taxids

    Returns:
        dict or None: Metadata dictionary from the last written batch, or None if nothing was written
    """
    (chunk, max_bins_per_seq, mess_true_file, mess_input_file, topk_taxa, min_tax_kmers, neg_extra, is_training) = args
    
    if is_training:
        return process_chunk_and_write(
            chunk, 
            max_bins_per_seq=max_bins_per_seq, 
            mess_true_file=mess_true_file, 
            mess_input_file=mess_input_file,
            topk_taxa=topk_taxa, 
            min_tax_kmers=min_tax_kmers, 
            neg_extra=neg_extra,
            is_training=is_training
        )
    else:
        return process_chunk_and_write(
            chunk, 
            max_bins_per_seq=max_bins_per_seq, 
            min_tax_kmers=min_tax_kmers,
            is_training=is_training
        )
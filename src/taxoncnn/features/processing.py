import re
import pandas as pd
import logging
import multiprocessing as mp
from alive_progress import alive_bar
from collections import defaultdict

from taxoncnn.features.features import compute_bin_features
from taxoncnn.utils.constants import CANONICAL_RANKS
import taxoncnn.utils.globals as globals
from taxoncnn.features.init import (
    _init_ncbi_private_db,
    effective_nprocs
)
from taxoncnn.utils.io_utils import (
    _write_rows_streaming_parquet,
    _write_rows_streaming_shards,
    prefetch
)
from taxoncnn.utils.tax_utils import (
    normalize_taxid,
    get_lineage_path,
    lineage_to_rank_map,
    predicted_rank
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


def build_tax_context(file_path, rows_per_chunk=1000, prefetch_buf=64, dispatch_batch=4, threads=0):
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
        _init_ncbi_private_db()
        with alive_bar(title="Building tax context", unknown="dots_waves") as bar:
            for chunk in reader:
                res = extract_tax_context_chunk(chunk)
                if res:
                    tax_context.update(res)
                bar()
        return tax_context

    logger.info(f"Using {nprocs} processes for tax_context build.")
    with mp.Pool(processes=nprocs, maxtasksperchild=200, initializer=_init_ncbi_private_db) as pool:
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


def process_chunk_iter(chunk, bin_size=1000, topk_taxa=None, min_tax_kmers=0, max_bins_per_seq=None,
                       mess_true_file=None, mess_input_file=None):
    """
    Process a chunk of classified sequences into per-(seq_id, taxon) rows with binned features and labels

    Args:
        chunk (pd.DataFrame): DataFrame with at least columns 'ID', 'Length', 'Kmers', 'Taxonomy'
        bin_size (int, optional): Number of k-mers per bin. Defaults to 1000
        topk_taxa (int or None, optional): If set, only process the top-k taxa per sequence. Defaults to None
        min_tax_kmers (int, optional): Minimum number of k-mers required for a taxon to be considered. Defaults to 0
        max_bins_per_seq (int or None, optional): Maximum number of bins per sequence (downsample if exceeded). Defaults to None
        mess_true_file (str or None, optional): Path to MESS true mapping file for ground truth taxids. Defaults to None
        mess_input_file (str or None, optional): Path to MESS input mapping file for predicted taxids. Defaults to None

    Yields:
        dict: Row dictionary with keys:
            'seq_id', 'taxon', 'true_taxon', 'bins', 'label', 'label_any',
            'label_rank', 'labels_per_rank', 'pred_rank', 'rank_index'
    """
    if chunk.empty:
        return
    view = chunk.loc[chunk['Classified'] == 'C', ['ID', 'Length', 'Kmers', 'Taxonomy']]
    if view.empty:
        logger.debug("No classified sequences in chunk, skipping.")
        return

    if mess_true_file and mess_input_file:
        if mess_true_file == mess_input_file:
            mess_map = pd.read_csv(mess_true_file, sep='\t', header=None, index_col=None, names=['seq_id', 'ref'])
            mess_map['tax_id'] = mess_map['ref'].str.split('|').str[1]
        else:
            try:
                logger.info(f"Processing MESS files: {mess_true_file}, {mess_input_file}")
                mess_df = pd.read_csv(mess_true_file, sep="\t", header=None, names=['seq_id', 'name'])
                mess_df['name'] = mess_df['name'].str.split('/').str[0].str[:-1]       
                mess_df["name"] = mess_df["name"].str.replace(
                    r'\.(\d)\d*(?:_.*)?',  # regex pattern
                    r'.\1',                # replacement
                    regex=True
                )     
                mess_input_df = pd.read_csv(mess_input_file, sep="\t", header=0)
                if (mess_input_df['tax_id'] == 0).all():
                    mess_input_df['tax_id'] = mess_input_df['fasta'].str.split('__').str[2]
                mess_input_df['fasta'] = mess_input_df['fasta'].str.split('__').str[-1]    
                mess_map = (mess_df.set_index('name')
                                .join(mess_input_df.set_index('fasta')[['tax_id']], how='left')
                                .reset_index()
                                .rename(columns={'index': 'name'}))        
            except:
                logger.warning(f"Error processing MESS files: {mess_true_file}, {mess_input_file}")

    for row in view.itertuples(index=False):
        seq_id, kmers_str = row.ID, row.Kmers
        if not isinstance(kmers_str, str) or not kmers_str:
            logger.debug(f"No k-mers for sequence {seq_id}, skipping.")
            continue
        
        if mess_true_file:
            try:
                true_tax_raw = mess_map.loc[mess_map['seq_id'] == seq_id, 'tax_id']
            except:
                logger.warning(f"Error retrieving true taxid for sequence {seq_id} from MESS map.")
                try:
                    true_tax_raw = row.ID.split('|')[1]
                except:
                    true_tax_raw = row.Taxonomy
            try:
                true_tax = normalize_taxid(int(true_tax_raw.iloc[0])) if not true_tax_raw.empty else normalize_taxid(row.Taxonomy)
            except:
                true_tax = normalize_taxid(int(true_tax_raw)) if true_tax_raw is not None else normalize_taxid(row.Taxonomy)
        else:
            try:
                true_tax_raw = row.ID.split('|')[1]
            except:
                true_tax_raw = row.Taxonomy
            true_tax = normalize_taxid(true_tax_raw)

        # True lineage + rank map (for Option B per-rank comparison)
        true_lineage = get_lineage_path(true_tax)
        if not true_lineage:
            continue
        true_at_rank = lineage_to_rank_map(true_lineage, CANONICAL_RANKS)

        # Accumulate per-bin counts for the sequence
        bin_counts_by_bin = {}
        tax_totals = defaultdict(int)
        cur_pos = 0

        for taxid, count in iter_kmer_tokens(kmers_str):
            tax_totals[taxid] += count
            cur_pos = add_to_bins(bin_counts_by_bin, bin_size, taxid, count, cur_pos)

        if not bin_counts_by_bin:
            continue

        # Candidate predicted taxa (filter by evidence)
        candidates = [t for t, c in tax_totals.items() if c >= min_tax_kmers]

        # Optional temporal downsample
        bin_indices = sorted(bin_counts_by_bin.keys())
        if max_bins_per_seq and len(bin_indices) > max_bins_per_seq:
            factor = len(bin_indices) / float(max_bins_per_seq)
            new_bins = {}
            for i, old_idx in enumerate(bin_indices):
                tgt = int(i / factor)
                d = new_bins.get(tgt)
                if d is None:
                    d = {}
                    new_bins[tgt] = d
                for taxid, cnt in bin_counts_by_bin[old_idx].items():
                    d[taxid] = d.get(taxid, 0) + cnt
            bin_counts_by_bin = new_bins
            bin_indices = sorted(bin_counts_by_bin.keys())

        # Build examples for each predicted taxon
        for pred_tax in candidates:
            pred_tax = normalize_taxid(pred_tax)

            pred_lineage = globals._shared_lineage_map.get(int(pred_tax), ())
            if not pred_lineage:
                continue

            # ----- Option B labels -----
            # any-lineage correctness (classic Option B)
            label_any = 1 if int(pred_tax) in true_lineage else 0

            # per-rank ancestor maps for predicted lineage
            pred_at_rank = lineage_to_rank_map(pred_lineage, CANONICAL_RANKS)

            # label at predicted rank only
            p_rank, p_idx = predicted_rank(pred_tax)
            if p_idx >= 0:
                label_rank = 1 if (true_at_rank[p_rank] is not None and true_at_rank[p_rank] == pred_at_rank[p_rank]) else 0
            else:
                label_rank = 0  # unknown rank -> treat as negative for rank-specific label

            # multi-head labels: match at every canonical rank
            labels_per_rank = []
            for r in CANONICAL_RANKS:
                tr = true_at_rank[r]
                pr = pred_at_rank[r]
                labels_per_rank.append(1 if (tr is not None and pr is not None and tr == pr) else 0)

            # 28-channel features for each bin relative to predicted lineage
            bins_vecs = []
            for b in bin_indices:
                kmer_tax_counts = bin_counts_by_bin[b]
                vec28 = compute_bin_features(kmer_tax_counts, pred_lineage, CANONICAL_RANKS)
                bins_vecs.append(vec28)

            yield {
                'seq_id': seq_id,
                'taxon': int(pred_tax),
                'true_taxon': int(true_tax),
                'bins': bins_vecs,
                # labels for Option B
                'label': label_any,               # legacy
                'label_any': label_any,
                'label_rank': label_rank,
                'labels_per_rank': labels_per_rank,
                'pred_rank': p_rank,
                'rank_index': p_idx,
            }


def process_chunk_and_write(chunk, max_bins_per_seq=None,
                            mess_true_file=None, mess_input_file=None):
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
    write_fmt  = globals._shared_write_format
    shard_size = globals._shared_shard_size
    target_len = globals._shared_target_length
    to_dtype   = globals._shared_to_dtype

    for row in process_chunk_iter(chunk, bin_size=1000, topk_taxa=8, min_tax_kmers=0, max_bins_per_seq=max_bins_per_seq,
                                  mess_true_file=mess_true_file, mess_input_file=mess_input_file):
        for_rows.append(row)
        need_flush = False
        if write_fmt == "parquet":
            need_flush = (len(for_rows) >= 512)
        else:
            need_flush = (len(for_rows) >= shard_size)

        if need_flush:
            logger.debug(f"Writing batch {batch_idx} with {len(for_rows)} rows.")
            if write_fmt == "parquet":
                meta = _write_rows_streaming_parquet(for_rows, max_batch_rows=256, use_half=False, quantize_u8=False)
            else:
                meta = _write_rows_streaming_shards(for_rows, max_batch_rows=shard_size,
                                                    target_length=target_len, to_dtype=to_dtype)
            logger.debug(f"Batch {batch_idx} written: {meta}")
            total_rows += meta.get('rows', 0) if meta else 0
            wrote_meta = meta
            for_rows.clear()
            batch_idx += 1

    if for_rows:
        logger.debug(f"Writing final batch {batch_idx} with {len(for_rows)} rows.")
        if write_fmt == "parquet":
            meta_tail = _write_rows_streaming_parquet(for_rows, max_batch_rows=256, use_half=False, quantize_u8=False)
        else:
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
    chunk, max_bins_per_seq, mess_true_file, mess_input_file = args
    return process_chunk_and_write(chunk, max_bins_per_seq=max_bins_per_seq, 
                                   mess_true_file=mess_true_file, mess_input_file=mess_input_file)
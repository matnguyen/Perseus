#!/usr/bin/env python3
"""
extract.py
====================================================

Stream Kraken output, produce per-(seq,taxon) 28-channel bin features,
and write either nested Parquet parts or channel-first .pt shards.

Usage (CLI):
    python extract.py <input> <output> [--format shards]
"""

import os
import pandas as pd
import argparse as ap
import logging
import multiprocessing as mp
from alive_progress import alive_bar
import pickle
import pyarrow.parquet as pq
from pathlib import Path
import gc
import glob
import json
from ete3 import NCBITaxa
import importlib
import shutil

import perseus.utils.globals as globals_mod
from perseus.utils.constants import CANONICAL_RANKS
from perseus.utils.tax_utils import (
    normalize_taxid,
    fetch_maps
)
from perseus.features.features import (
    _torch_dtype,
    _resample_TN_to_T
)
from perseus.features.init import (
    init_worker,
    effective_nprocs,
    _next_worker_part_name
)
from perseus.features.processing import (
    build_tax_context
)
from perseus.features.features import compute_bin_features
from perseus.utils.constants import CANONICAL_RANKS
import perseus.utils.globals as globals
from perseus.features.init import (
    _init_ncbi_private_db,
    effective_nprocs
)
from perseus.utils.io_utils import (
    prefetch
)
from perseus.utils.tax_utils import (
    normalize_taxid,
    get_lineage_path,
    lineage_to_rank_map,
    predicted_rank
)
from perseus.utils.targets import (
    compute_cutoff_and_exclusion,
    build_targets_from_cutoff
)
from perseus.features.processing import (
    iter_kmer_tokens,
    add_to_bins
)

import numpy as np
from collections import defaultdict
import torch

# --- Kill hidden thread oversubscription (BLAS/numexpr/etc.) ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_chunk_iter(chunk, bin_size=1000, topk_taxa=None, min_tax_kmers=0, max_bins_per_seq=None,
                       mess_true_file=None, mess_input_file=None, world=None, excluded_train_sets=None):
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
                
            # Compute cutoff & targets
            if world is not None and excluded_train_sets is not None:          
                cutoff_name, cutoff_idx, is_exc, exc_level = compute_cutoff_and_exclusion(true_at_rank, world, excluded_train_sets)
                targets = build_targets_from_cutoff(cutoff_idx)

            yield {
                'seq_id': seq_id,
                'taxon': int(pred_tax),
                'true_taxon': int(true_tax),
                'bins': bins_vecs,
                'n_bins': len(bins_vecs),
                # labels for Option B
                'label': label_any,               # legacy
                'label_any': label_any,
                'label_rank': label_rank,
                'labels_per_rank': labels_per_rank,
                'pred_rank': p_rank,
                'rank_index': p_idx,
                'targets': targets if world is not None and excluded_train_sets is not None else None,
                'cutoff_rank_index': int(cutoff_idx),
                'is_excluded': bool(is_exc),
                'excluded_level': exc_level
            }


def read_kraken_file_for_training(file_path: str,
                                  output_path: str,
                                  world: str,
                                  excl_species_file: str,
                                  excl_genera_file: str,
                                  excl_families_file: str,
                                  mess_truth_file: str = None,
                                  mess_input_file: str = None,
                                  chunksize: int = 1000,
                                  threads: int = 0,
                                  write_format: str = "shards",
                                  shard_size: int = 4096,
                                  target_length: int = 1024,
                                  to_dtype: str = "float32",
                                  max_bins_per_seq: int = None):
    """
    Training-specific extraction: writes shards/parquet that include:
      - 'targets' (list[int] or per-rank columns) in coarse->fine order,
      - 'cutoff_rank_index' (int),
      - 'is_excluded' (bool),
      - 'excluded_level' (str)

    world: "CORE", "S", "G", "F"  (controls cutoffs)
    """
    logger.debug(f"Starting read_kraken_file with file_path={file_path}, output_path={output_path}, chunksize={chunksize}, format={write_format}")
    
    # load excluded sets once
    excl_sets = {
       "species": set() if not excl_species_file else {line.strip() for line in open(excl_species_file)},
       "genera": set() if not excl_genera_file else {line.strip() for line in open(excl_genera_file)},
       "families": set() if not excl_families_file else {line.strip() for line in open(excl_families_file)},
    }

    # 1) Build/load tax_context
    tax_context_path = output_path + ".tax_context.pkl"
    if os.path.exists(tax_context_path):
        logger.info(f"Loading cached tax_context from {tax_context_path}")
        with open(tax_context_path, "rb") as f:
            tax_context = pickle.load(f)
        logger.debug(f"Loaded tax_context with {len(tax_context)} entries")
    else:
        logger.info(f"Building tax_context from {file_path}")
        tax_context = build_tax_context(file_path, rows_per_chunk=2000, prefetch_buf=64, dispatch_batch=6, threads=threads)
        with open(tax_context_path, "wb") as f:
            pickle.dump(tax_context, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved tax_context to {tax_context_path}")
        logger.debug(f"Built tax_context with {len(tax_context)} entries")

    # 2) Collect all numeric taxids
    all_taxids = set()
    for _, counts in tax_context.items():
        for t in counts.keys():
            try:
                all_taxids.add(normalize_taxid(int(t)))
            except ValueError:
                logger.debug(f"Skipping non-numeric taxid: {t}")
                continue
    logger.debug(f"Collected {len(all_taxids)} unique numeric taxids")

    # 3) Precompute/load lineage/descendant/canonical maps
    lineage_map_path    = output_path + ".lineage_map.pkl"
    descendant_map_path = output_path + ".descendant_map.pkl"
    canonical_map_path  = output_path + ".canonical_map.pkl"

    if os.path.exists(lineage_map_path) and os.path.exists(descendant_map_path) and os.path.exists(canonical_map_path):
        logger.info("Loading cached lineage/descendant/canonical maps.")
        with open(lineage_map_path, "rb") as f:
            lineage_map = pickle.load(f)
        with open(descendant_map_path, "rb") as f:
            descendant_map = pickle.load(f)
        with open(canonical_map_path, "rb") as f:
            canonical_map = pickle.load(f)
        logger.debug(f"Loaded lineage_map ({len(lineage_map)}), descendant_map ({len(descendant_map)}), canonical_map ({len(canonical_map)})")
    else:
        lineage_map, descendant_map, canonical_map = {}, {}, {}
        if threads == 0:
            nprocs = effective_nprocs()
            logger.info(f"Precomputing maps for {len(all_taxids)} taxids using {nprocs} processes.")
            with mp.Pool(processes=nprocs, maxtasksperchild=200) as pool:
                for tid, lineage, descendants, canonicals in pool.imap_unordered(fetch_maps, all_taxids, chunksize=chunksize):
                    lineage_map[tid]    = lineage
                    descendant_map[tid] = descendants
                    canonical_map[tid]  = canonicals
        elif threads == 1:
            logger.info(f"Precomputing maps for {len(all_taxids)} taxids using single-threaded mode.")
            for tid in all_taxids:
                tid, lineage, descendants, canonicals = fetch_maps(tid)
                lineage_map[tid]    = lineage
                descendant_map[tid] = descendants
                canonical_map[tid]  = canonicals
        else:
            nprocs = threads
            logger.info(f"Precomputing maps for {len(all_taxids)} taxids using {nprocs} processes (user-specified).")
            with mp.Pool(processes=nprocs, maxtasksperchild=200) as pool:
                for tid, lineage, descendants, canonicals in pool.imap_unordered(fetch_maps, all_taxids, chunksize=chunksize):
                    lineage_map[tid]    = lineage
                    descendant_map[tid] = descendants
                    canonical_map[tid]  = canonicals

        with open(lineage_map_path, "wb") as f:
            pickle.dump(lineage_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(descendant_map_path, "wb") as f:
            pickle.dump(descendant_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(canonical_map_path, "wb") as f:
            pickle.dump(canonical_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved lineage/descendant/canonical maps.")

    # 4) Process CSV in parallel, writing outputs in workers
    out_dir = Path(output_path)
    if write_format == "parquet" and out_dir.suffix == ".parquet":
        out_dir = out_dir.with_suffix("")  # "foo.parquet" -> "foo/"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory: {str(out_dir)} (format={write_format})")

    rows_per_df = 2000
    nprocs = effective_nprocs()
    logger.info(f"Processing with {nprocs if threads==0 else threads} workers; writing under {str(out_dir)}")

    wrote_rows = 0
    wrote_files = 0

    # Shared manifest list for shards
    manager = mp.Manager()
    manifest_paths = manager.list() if write_format == "shards" else None

    with pd.read_csv(
        file_path, sep='\t', header=None,
        names=['Classified', 'ID', 'Taxonomy', 'Length', 'Kmers'],
        dtype={'Classified': 'category', 'ID': 'string', 'Taxonomy': 'string', 'Length': 'int32', 'Kmers': 'string'},
        engine='c',
        chunksize=rows_per_df
    ) as reader:

        if threads == 0:
            with mp.Pool(
                processes=nprocs,
                initializer=init_worker,
                initargs=(tax_context, lineage_map, descendant_map, canonical_map, str(out_dir),
                          write_format, shard_size, target_length, to_dtype, manifest_paths),
                maxtasksperchild=200
            ) as pool:
                results = pool.imap_unordered(
                    process_chunk_and_write_wrapper,
                    ((chunk, max_bins_per_seq, mess_truth_file, mess_input_file, world, excl_sets) for chunk in reader),
                    chunksize=1
                )
                with alive_bar(title="Processing chunks", unknown="dots_waves") as bar:
                    for meta in results:
                        if meta:
                            wrote_rows  += int(meta.get('rows', 0))
                            wrote_files += 1
                        bar()
                        gc.collect()

        elif threads == 1:
            init_worker(tax_context, lineage_map, descendant_map, canonical_map, str(out_dir),
                        write_format, shard_size, target_length, to_dtype, manifest_paths)
            for chunk in reader:
                meta = process_chunk_and_write(chunk, max_bins_per_seq=max_bins_per_seq,
                                               mess_true_file=mess_truth_file, mess_input_file=mess_input_file, world=world, excluded_train_sets=excl_sets)
                if meta:
                    wrote_rows  += int(meta.get('rows', 0))
                    wrote_files += 1
                gc.collect()
        else:
            with mp.Pool(
                processes=threads,
                initializer=init_worker,
                initargs=(tax_context, lineage_map, descendant_map, canonical_map, str(out_dir),
                          write_format, shard_size, target_length, to_dtype, manifest_paths),
                maxtasksperchild=200
            ) as pool:
                results = pool.imap_unordered(
                    process_chunk_and_write_wrapper,
                    ((chunk, max_bins_per_seq, mess_truth_file, mess_input_file, world, excl_sets) for chunk in reader),
                    chunksize=1
                )
                with alive_bar(title="Processing chunks", unknown="dots_waves") as bar:
                    for meta in results:
                        if meta:
                            wrote_rows  += int(meta.get('rows', 0))
                            wrote_files += 1
                        bar()
                        gc.collect()

    logger.info(f"Wrote {wrote_files} part files (~{wrote_rows} rows) under {str(out_dir)}")

    # If shards: write manifest json (for your shard trainer)
    if write_format == "shards":
        mani = {
            "source": str(file_path),
            "outputs": list(manifest_paths) if manifest_paths is not None else [],
            "channels": 28,
            "target_length": int(target_length),
            "dtype": str(to_dtype),
            "shard_size": int(shard_size),
            "counts": {"approx_rows": int(wrote_rows), "files": int(wrote_files)},
            "labels": {
                "y_any": "pred_tax ∈ true_lineage",
                "y_rank": "pred ancestor at predicted rank equals true ancestor at that rank",
                "y_per_rank": f"length {len(CANONICAL_RANKS)}; equality per canonical rank",
                "rank_index": f"index in CANONICAL_RANKS: {CANONICAL_RANKS}"
            }
        }
        mani_path = out_dir / "permute_manifest.json"
        with open(mani_path, "w") as f:
            json.dump(mani, f, indent=2)
        logger.info(f"Wrote shard manifest → {mani_path}")
    else:
        logger.info("Done. You can read later via pyarrow.dataset.dataset(str(out_dir)).to_table()")


def process_chunk_and_write(chunk, max_bins_per_seq=None,
                            mess_true_file=None, mess_input_file=None, world=None, excluded_train_sets=None):
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
                                  mess_true_file=mess_true_file, mess_input_file=mess_input_file, world=world, excluded_train_sets=excluded_train_sets):
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
    chunk, max_bins_per_seq, mess_true_file, mess_input_file, world, excluded_train_sets = args
    return process_chunk_and_write(chunk, max_bins_per_seq=max_bins_per_seq, 
                                   mess_true_file=mess_true_file, mess_input_file=mess_input_file,
                                   world=world, excluded_train_sets=excluded_train_sets)

def combine_parquet_parts(parts_dir, output_file, pattern="part-*.parquet",
                          row_group_size=None, compression="zstd",
                          write_statistics=True, cleanup_parts=False,
                          sort_files=True):
    """
    Combine many small Parquet part files (with nested 'bins') into one file

    Args:
        parts_dir (str or Path): Directory containing Parquet part files
        output_file (str or Path): Path to write the combined Parquet file
        pattern (str, optional): Glob pattern for part files. Defaults to "part-*.parquet"
        row_group_size (int or None, optional): Row group size for output file. Defaults to None
        compression (str, optional): Compression algorithm. Defaults to "zstd"
        write_statistics (bool, optional): Whether to write statistics. Defaults to True
        cleanup_parts (bool, optional): Whether to remove part files after combining. Defaults to False
        sort_files (bool, optional): Whether to sort part files before combining. Defaults to True

    Returns:
        None
    """
    parts_dir   = str(parts_dir)
    output_file = str(output_file)

    pd_path = Path(parts_dir).resolve()
    of_path = Path(output_file).resolve()

    if (of_path.exists() and of_path.is_dir()) or (of_path == pd_path):
        of_path = pd_path / "combined.parquet"
    elif output_file.endswith(os.sep) or output_file.endswith("\\"):
        of_path = pd_path / "combined.parquet"

    if of_path.suffix.lower() not in (".parquet", ".parq"):
        if of_path.exists() and of_path.is_dir():
            of_path = of_path / "combined.parquet"

    files = glob.glob(os.path.join(parts_dir, pattern))
    if not files:
        raise FileNotFoundError(f"No Parquet parts matching '{pattern}' in {parts_dir}")
    if sort_files:
        files.sort()

    first_pf = pq.ParquetFile(files[0])
    target_schema = first_pf.schema_arrow

    of_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Combining {len(files)} parts from {parts_dir} → {str(of_path)}")

    writer = pq.ParquetWriter(
        str(of_path),
        target_schema,
        compression=compression,
        use_dictionary=True,
        write_statistics=write_statistics,
    )

    wrote_rows = 0
    try:
        with alive_bar(len(files), title="Combining Parquet parts") as bar:
            for path in files:
                pf = pq.ParquetFile(path)
                for rg_idx in range(pf.num_row_groups):
                    tbl = pf.read_row_group(rg_idx)
                    if not tbl.schema.equals(target_schema, check_metadata=False):
                        try:
                            tbl = tbl.cast(target_schema, safe=False)
                        except Exception as e:
                            writer.close()
                            raise TypeError(
                                f"Schema mismatch reading '{path}' (row-group {rg_idx}).\n"
                                f"Expected {target_schema}, got {tbl.schema}.\n"
                                f"Casting error: {e}"
                            ) from e
                    writer.write_table(tbl, row_group_size=row_group_size)
                    wrote_rows += tbl.num_rows
                bar()
    finally:
        writer.close()

    logger.info(f"Wrote {wrote_rows} rows to {str(of_path)}")

    if cleanup_parts:
        removed = 0
        for path in files:
            try:
                os.remove(path); removed += 1
            except OSError:
                pass
        logger.info(f"Removed {removed}/{len(files)} part files")
        
def _write_rows_streaming_shards(rows, max_batch_rows=4096, target_length=1024, to_dtype="float32"):
    """
    Writes a shard .pt containing:
        x: [N, C=28, T]  (resampled to target_length if >0; else pad to max T in shard)
        y: [N] (float32, legacy mirror of y_any)
        y_any: [N] (float32)
        y_rank: [N] (float32)
        y_per_rank: [N, R] (float32, R=len(CANONICAL_RANKS))
        rank_index: [N] (int8)
        seq_id (list[str]), taxon (list[int]), lengths: [N] (if target_length==0)

    Args:
        rows (Iterable[dict]): Feature rows to write. Iterable of {
            'seq_id': str, 'taxon': int, 'bins': list[list[float]],
            'label_any': int, 'label_rank': int, 'labels_per_rank': list[int],
            'pred_rank': str|None, 'rank_index': int
        }
        max_batch_rows (int, optional): Max rows per batch. Defaults to 4096
        target_length (int, optional): Resample to this length if >0. Defaults to 1024
        to_dtype (str, optional): Target dtype for x. Defaults to "float32"

    Returns:
        dict or None: Returns dict with number of rows written and filename, or None if no rows
    """
    if not rows:
        return None

    C = 22
    R = len(CANONICAL_RANKS)
    fname = _next_worker_part_name("pt")
    fpath = os.path.join(globals._shared_out_dir, fname)

    X_list, y_any_list, y_rank_list, y_per_rank_list = [], [], [], []
    id_list, tax_list, true_tax_list, len_list, rank_index_list = [], [], [], [], []
    targets_list, cutoff_rank_index_list, is_excluded_list, excluded_level_list = [], [], [], []
    T_max = 0
    tl = int(target_length)
    for r in rows:
        arr = np.asarray(r['bins'], dtype=np.float32)  # [T,28]
        if arr.ndim == 1: arr = arr[0:1]
        if arr.shape[1] != C:
            continue
        if tl > 0 and arr.shape[0] != tl:
            arr = _resample_TN_to_T(arr, tl)
        T = arr.shape[0]
        T_max = max(T_max, T)
        X_list.append(torch.from_numpy(arr.T))   # [28,T]

        # labels
        y_any_list.append(float(r.get('label_any', 0)))
        y_rank_list.append(float(r.get('label_rank', 0)))
        lpr = r.get('labels_per_rank', [0]*R)
        if len(lpr) != R:
            tmp = np.zeros(R, dtype=np.float32)
            m = min(R, len(lpr))
            tmp[:m] = np.asarray(lpr[:m], dtype=np.float32)
            lpr = tmp.tolist()
        y_per_rank_list.append(torch.tensor(lpr, dtype=torch.float32))

        id_list.append(str(r.get('seq_id', "")))
        tax_list.append(int(r.get('taxon', -1)))
        true_tax_list.append(int(r.get('true_taxon', -1)))  # <--- ADD THIS LINE
        rank_index_list.append(int(r.get('rank_index', -1)))
        len_list.append(T)
        
        targets_list.append(r.get('targets', None))
        cutoff_rank_index_list.append(int(r.get('cutoff_rank_index', -1)))
        is_excluded_list.append(bool(r.get('is_excluded', False)))
        excluded_level_list.append(str(r.get('excluded_level', "")))
        
        if len(X_list) >= max_batch_rows:
            break

    if not X_list:
        return None

    # Pad or stack
    dt = _torch_dtype(to_dtype)
    if tl > 0:
        T_final = tl
        x = torch.stack([xi.to(dt) for xi in X_list], dim=0)  # [N,28,T]
    else:
        T_final = T_max
        x = torch.zeros(len(X_list), C, T_final, dtype=dt)
        for i, xi in enumerate(X_list):
            t = xi.shape[-1]
            x[i, :, :t] = xi.to(dt)

    y_any  = torch.tensor(y_any_list, dtype=torch.float32)
    y_rank = torch.tensor(y_rank_list, dtype=torch.float32)
    y_pr   = torch.stack(y_per_rank_list, dim=0)  # [N,R]
    # legacy mirror
    y_legacy = y_any.clone()

    bundle = {
        "x": x,
        # "y": y_legacy,              # legacy
        # "y_any": y_any,
        # "y_rank": y_rank,
        "y_per_rank": y_pr,
        "seq_id": id_list,
        "taxon": tax_list,
        "true_taxon": true_tax_list,  
        "rank_index": torch.tensor(rank_index_list, dtype=torch.int8),
        "target": torch.tensor(targets_list, dtype=torch.int8) if targets_list[0] is not None else None,
        "cutoff_rank_index": torch.tensor(cutoff_rank_index_list, dtype=torch.int8),
        "is_excluded": torch.tensor(is_excluded_list, dtype=torch.bool),
        "excluded_level": excluded_level_list,
    }
    if tl <= 0:
        bundle["lengths"] = torch.tensor(len_list, dtype=torch.int32)

    # Write shard
    torch.save(bundle, fpath)

    # record path into shared manifest list (thread/process-safe)
    try:
        if globals._shared_manifest_paths is not None:
            globals._shared_manifest_paths.append(fpath)
    except Exception:
        pass

    return {'rows': len(X_list), 'file': fname}


if __name__ == '__main__':
    """
    Command-line interface for chunked and parallel processing of Kraken output with Option-B labeling

    Parses arguments, runs extraction, and optionally combines Parquet parts
    """
    parser = ap.ArgumentParser(description='Chunked and parallel processing of Kraken output with Option-B labeling.')
    parser.add_argument('file_path', type=str, help='Path to the Kraken output file')
    parser.add_argument('output_path', type=str, help='Path to save the processed output (dir or .parquet)')

    # unchanged
    parser.add_argument('--rows-per-chunk', type=int, default=20000, help='Rows per DataFrame chunk for pools')
    parser.add_argument('--threads', type=int, default=0, help='Number of worker processes (0=auto)')
    parser.add_argument('--max-bins-per-seq', type=int, default=None, help='Max bins per (seq_id, taxon) (default: None)')

    # NEW: output format and shard controls
    parser.add_argument('--format', choices=['parquet','shards'], default='parquet',
                        help='Output format: parquet (original) or shards (.pt, channel-first)')
    parser.add_argument('--shard-size', type=int, default=4096, help='Samples per shard (.pt)')
    parser.add_argument('--target-length', type=int, default=1024,
                        help='Resample time to this length for shards (0 = pad to shard max)')
    parser.add_argument('--to-dtype', choices=['float32','float16','bfloat16'], default='float32',
                        help='Stored dtype for shard tensor')
    parser.add_argument('--mess-truth-file', type=str, default=None,
                        help='(Not used) Path to MESS truth file for Option-B labeling (currently inferred from IDs)')
    parser.add_argument('--mess-input-file', type=str, default=None,
                        help='(Not used) Path to MESS input file for Option-B labeling (currently inferred from IDs)')
    parser.add_argument('--world', type=str, choices=['CORE','S','G','F'], default=None,
                        help='World setting for cutoff computation (if using excluded sets)')
    parser.add_argument('--excl-species-file', type=str, default=None,
                        help='Path to file with excluded species taxids (one per line)')
    parser.add_argument('--excl-genera-file', type=str, default=None,
                        help='Path to file with excluded genera taxids (one per line)')
    parser.add_argument('--excl-families-file', type=str, default=None,
                        help='Path to file with excluded families taxids (one per line)')

    args = parser.parse_args()
    
    if args.threads == 1:
        globals_mod = importlib.import_module("perseus.utils.globals")
        globals_mod.NCBI = NCBITaxa() # Pre-initialize for single-threaded mode
    
    if args.mess_truth_file and args.mess_input_file is None:
        logger.critical("MESS truth file provided without MESS input file; this is required. Exiting.")
        raise SystemExit("MESS truth file provided without MESS input file; required pair missing.")
    if args.mess_input_file and args.mess_truth_file is None:
        logger.critical("MESS input file provided without MESS truth file; this is required. Exiting.")
        raise SystemExit("MESS input file provided without MESS truth file; required pair missing.")

    # Run extraction
    read_kraken_file_for_training(
        args.file_path, args.output_path,
        chunksize=1000, threads=args.threads, max_bins_per_seq=args.max_bins_per_seq,
        write_format=args.format, shard_size=args.shard_size,
        target_length=args.target_length, to_dtype=args.to_dtype,
        mess_truth_file=args.mess_truth_file,
        mess_input_file=args.mess_input_file,
        world=args.world,
        excl_species_file=args.excl_species_file,
        excl_genera_file=args.excl_genera_file,
        excl_families_file=args.excl_families_file,
    )

    # Parquet-only combining
    if args.format == "parquet":
        out_p = Path(args.output_path)
        if out_p.suffix.lower() == ".parquet":
            parts_dir  = str(out_p.with_suffix(""))     # strip '.parquet' → directory with parts
            final_file = str(out_p)                     # exact file requested by the user
        else:
            parts_dir  = str(out_p)                     # treat as directory of parts (e.g., '.parquet3')
            final_file = str(out_p / "combined.parquet")

        try:
            combine_parquet_parts(
                parts_dir=parts_dir,
                output_file=final_file,
                row_group_size=256_000,
                compression="zstd",
                write_statistics=True,
                cleanup_parts=False,
                sort_files=True,
            )
            logger.info(f"Combined parts under {parts_dir} → {final_file}")
        except Exception as e:
            logger.exception(f"Failed to combine parts from {parts_dir}: {e}")
    else:
        logger.info("Shards written; no Parquet combine step.")
        
    # Cleanup ETE3 temp dirs
    for tmpdir in glob.glob("/tmp/perseus_ete3db_*"):
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
            logger.debug(f"Deleted temp dir: {tmpdir}")

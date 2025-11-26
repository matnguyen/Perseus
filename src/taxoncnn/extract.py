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

from taxoncnn.utils.constants import CANONICAL_RANKS
from taxoncnn.utils.tax_utils import (
    normalize_taxid,
    fetch_maps
)
from taxoncnn.features.init import (
    init_worker,
    effective_nprocs
)
from taxoncnn.features.processing import (
    process_chunk_and_write,
    process_chunk_and_write_wrapper,
    build_tax_context
)

# --- Kill hidden thread oversubscription (BLAS/numexpr/etc.) ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_kraken_file(file_path, output_path, chunksize=5000, threads=0, max_bins_per_seq=None,
                     write_format="parquet", shard_size=4096, target_length=1024, to_dtype="float32",
                     mess_truth_file=None, mess_input_file=None):
    """
    Stream Kraken output, produce per-(seq, taxon) 28-channel bin features,
    and write either nested Parquet parts or channel-first .pt shards

    Args:
        file_path (str): Path to the Kraken output file.
        output_path (str): Path to save the processed output (directory or .parquet file)
        chunksize (int, optional): Number of rows per DataFrame chunk for pools. Defaults to 5000
        threads (int, optional): Number of worker processes (0=auto, 1=single-threaded). Defaults to 0
        max_bins_per_seq (int or None, optional): Max bins per (seq_id, taxon). Defaults to None
        write_format (str, optional): Output format: 'parquet' or 'shards'. Defaults to 'parquet'
        shard_size (int, optional): Samples per shard (.pt). Defaults to 4096
        target_length (int, optional): Resample time to this length for shards (0 = pad to shard max). Defaults to 1024
        to_dtype (str, optional): Stored dtype for shard tensor. Defaults to "float32"
        mess_truth_file (str or None, optional): Path to MESS truth file for Option-B labeling. Defaults to None
        mess_input_file (str or None, optional): Path to MESS input file for Option-B labeling. Defaults to None

    Returns:
        None
    """
    logger.debug(f"Starting read_kraken_file with file_path={file_path}, output_path={output_path}, chunksize={chunksize}, format={write_format}")

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
                    ((chunk, max_bins_per_seq, mess_truth_file, mess_input_file) for chunk in reader),
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
                                               mess_true_file=mess_truth_file, mess_input_file=mess_input_file)
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
                    ((chunk, max_bins_per_seq, mess_truth_file, mess_input_file) for chunk in reader),
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

    args = parser.parse_args()
    
    if args.threads == 1:
        globals_mod = importlib.import_module("taxoncnn.utils.globals")
        globals_mod.NCBI = NCBITaxa() # Pre-initialize for single-threaded mode
    
    if args.mess_truth_file and args.mess_input_file is None:
        logger.critical("MESS truth file provided without MESS input file; this is required. Exiting.")
        raise SystemExit("MESS truth file provided without MESS input file; required pair missing.")
    if args.mess_input_file and args.mess_truth_file is None:
        logger.critical("MESS input file provided without MESS truth file; this is required. Exiting.")
        raise SystemExit("MESS input file provided without MESS truth file; required pair missing.")

    # Run extraction
    read_kraken_file(
        args.file_path, args.output_path,
        chunksize=1000, threads=args.threads, max_bins_per_seq=args.max_bins_per_seq,
        write_format=args.format, shard_size=args.shard_size,
        target_length=args.target_length, to_dtype=args.to_dtype,
        mess_truth_file=args.mess_truth_file,
        mess_input_file=args.mess_input_file
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

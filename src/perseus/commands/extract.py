#!/usr/bin/env python3
import os
import pandas as pd
import argparse as ap
import logging
import multiprocessing as mp
from alive_progress import alive_bar
import gc
import math
import glob
import json
from ete3 import NCBITaxa
import shutil
from pathlib import Path

import perseus.utils.globals as globals_mod
from perseus.utils.constants import CANONICAL_RANKS, N_CHANNELS
from perseus.utils.tax_utils import (
    normalize_taxid,
    fetch_maps
)
from perseus.features.init import (
    init_worker,
    effective_nprocs
)
from perseus.features.processing import (
    process_chunk_and_write,
    process_chunk_and_write_wrapper,
    build_tax_context
)

# --- Kill hidden thread oversubscription (BLAS/numexpr/etc.) ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

LOG = logging.getLogger(__name__)

def read_kraken_file(
        file_path,
        output_path, 
        rows_per_chunk=5000, 
        threads=0, 
        max_bins_per_seq=None,
        shard_size=4096, 
        target_length=1024, 
        to_dtype="float32",
        min_tax_kmers=10
    ):
    LOG.info("Starting feature extraction...")
    LOG.info("Input file: %s", file_path)
    LOG.info("Output directory: %s", output_path)
    LOG.info("Threads: %d", threads if threads > 0 else effective_nprocs())
    LOG.info("Minimum k-mers per taxon: %d", min_tax_kmers)
    LOG.debug("Starting read_kraken_file with file_path=%s, output_path=%s, rows_per_chunk=%d", file_path, output_path, rows_per_chunk)
    
    # Set vars needed only for training
    mess_true_file = None
    mess_input_file = None
    topk_taxa = None
    neg_extra = None
    is_training = False

    # Build tax_context
    LOG.info("Precomputing sequence → taxid k-mer count map from %s", file_path)
    tax_context = build_tax_context(file_path, rows_per_chunk=rows_per_chunk, prefetch_buf=64, dispatch_batch=6, threads=threads)
    
    if len(tax_context) == 0:
        LOG.error("No valid taxonomic evidence could be built from the input file. Please check the file format and contents.")
        raise SystemExit(1)
    
    LOG.debug("Built taxonomic context: %d sequences with aggregated k-mer taxid counts", len(tax_context))

    # Collect all numeric taxids
    LOG.info("Collecting unique numeric taxids...")
    all_taxids = set()
    for _, counts in tax_context.items():
        for t in counts.keys():
            try:
                all_taxids.add(normalize_taxid(int(t)))
            except ValueError:
                LOG.debug("Skipping non-numeric taxid: %s", t)
                continue    
    LOG.debug("Collected %d unique numeric taxids", len(all_taxids))

    lineage_map, descendant_map, canonical_map = {}, {}, {}
    if threads == 0:
        nprocs = effective_nprocs()
        LOG.info("Precomputing lineage/descendant maps for %d taxids using %d processes", len(all_taxids), nprocs)
        with mp.Pool(processes=nprocs, maxtasksperchild=200) as pool:
            for tid, lineage, descendants, canonicals in pool.imap_unordered(fetch_maps, all_taxids, chunksize=rows_per_chunk):
                lineage_map[tid]    = lineage
                descendant_map[tid] = descendants
                canonical_map[tid]  = canonicals
    elif threads == 1:
        LOG.info("Precomputing lineage/descendant maps for %d taxids using single-threaded mode", len(all_taxids))
        for tid in all_taxids:
            tid, lineage, descendants, canonicals = fetch_maps(tid)
            lineage_map[tid]    = lineage
            descendant_map[tid] = descendants
            canonical_map[tid]  = canonicals
    else:
        nprocs = threads
        LOG.info("Precomputing lineage/descendant maps for %d taxids using %d processes (user-specified)", len(all_taxids), nprocs)
        with mp.Pool(processes=nprocs, maxtasksperchild=200) as pool:
            for tid, lineage, descendants, canonicals in pool.imap_unordered(fetch_maps, all_taxids, chunksize=rows_per_chunk):
                lineage_map[tid]    = lineage
                descendant_map[tid] = descendants
                canonical_map[tid]  = canonicals
    LOG.info("Completed precomputing taxonomic maps")
    
    # Process CSV in parallel, writing outputs in workers
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.debug(f"Output directory: {str(out_dir)}")

    nprocs = effective_nprocs() if threads == 0 else threads
    LOG.info(f"Processing with %d workers; writing under %s", nprocs if threads==0 else threads, str(out_dir))
    
    # Count total rows for better chunk size calculation
    LOG.info("Counting rows in input file...")
    with open(file_path, "r") as fh:
        total_rows = sum(1 for _ in fh)
    LOG.info(f"Total rows in file: %d", total_rows)
    
    # Adjust chunksize based on number of workers 
    adjusted_chunksize = max(total_rows // (nprocs * 10), rows_per_chunk)
    n_chunks = math.ceil(total_rows / adjusted_chunksize)
    LOG.debug("Adjusted chunksize for reading: %d", adjusted_chunksize)

    wrote_rows = 0
    wrote_files = 0

    # Shared manifest list for shards
    manager = mp.Manager()
    manifest_paths = manager.list() 

    LOG.info("Starting chunked processing of Kraken output...")
    with pd.read_csv(
        file_path, sep='\t', header=None,
        names=['Classified', 'ID', 'Taxonomy', 'Length', 'Kmers'],
        dtype={'Classified': 'category', 'ID': 'string', 'Taxonomy': 'string', 'Length': 'int32', 'Kmers': 'string'},
        engine='c',
        chunksize=adjusted_chunksize
    ) as reader:

        if threads == 0:
            with mp.Pool(
                processes=nprocs,
                initializer=init_worker,
                initargs=(tax_context, lineage_map, descendant_map, canonical_map, str(out_dir),
                          shard_size, target_length, to_dtype, manifest_paths),
                maxtasksperchild=200
            ) as pool:
                results = pool.imap_unordered(
                    process_chunk_and_write_wrapper,
                    ((chunk, max_bins_per_seq, mess_true_file, mess_input_file, topk_taxa, min_tax_kmers, neg_extra, is_training) for chunk in reader),
                    chunksize=1
                )
                with alive_bar(n_chunks, title="Processing chunks") as bar:
                    for meta in results:
                        if meta:
                            wrote_rows  += int(meta.get('rows', 0))
                            wrote_files += 1
                        bar()
                        gc.collect()

        elif threads == 1:
            init_worker(tax_context, lineage_map, descendant_map, canonical_map, str(out_dir),
                        shard_size, target_length, to_dtype, manifest_paths)
            for chunk in reader:
                meta = process_chunk_and_write(
                    chunk,
                    mess_true_file=mess_true_file,
                    mess_input_file=mess_input_file,
                    topk_taxa=topk_taxa, 
                    max_bins_per_seq=max_bins_per_seq,
                    min_tax_kmers=min_tax_kmers,
                    neg_extra=neg_extra,
                    is_training=is_training
                )
                if meta:
                    wrote_rows  += int(meta.get('rows', 0))
                    wrote_files += 1
                gc.collect()
        else:
            with mp.Pool(
                processes=threads,
                initializer=init_worker,
                initargs=(tax_context, lineage_map, descendant_map, canonical_map, str(out_dir),
                          shard_size, target_length, to_dtype, manifest_paths),
                maxtasksperchild=200
            ) as pool:
                results = pool.imap_unordered(
                    process_chunk_and_write_wrapper,
                    ((chunk, max_bins_per_seq, mess_true_file, mess_input_file, topk_taxa, min_tax_kmers, neg_extra, is_training) for chunk in reader),
                    chunksize=1
                )
                with alive_bar(n_chunks, title="Processing chunks") as bar:
                    for meta in results:
                        if meta:
                            wrote_rows  += int(meta.get('rows', 0))
                            wrote_files += 1
                        bar()
                        gc.collect()

    LOG.info(f"Wrote %d part files (~%d rows) under %s", wrote_files, wrote_rows, str(out_dir))
    
    # Write manifest json 
    mani = {
        "source": str(file_path),
        "outputs": list(manifest_paths) if manifest_paths is not None else [],
        "channels": N_CHANNELS,
        "target_length": int(target_length),
        "dtype": str(to_dtype),
        "shard_size": shard_size,
        "topk_taxa": topk_taxa,
        "min_tax_kmers": min_tax_kmers,
        "neg_extra": neg_extra,
        "counts": {"approx_rows": int(wrote_rows), "files": int(wrote_files)},
        "labels": {
            "labels_per_rank": f"length {len(CANONICAL_RANKS)}; equality per canonical rank",
            "rank_index": f"index in CANONICAL_RANKS: {CANONICAL_RANKS}"
        }
    }
    mani_path = out_dir / "manifest.json"
    with open(mani_path, "w") as f:
        json.dump(mani, f, indent=2)
    LOG.info("Wrote shard manifest → %s", mani_path)

def main():
    parser = ap.ArgumentParser(description='Chunked and parallel processing of Kraken output with Option-B labeling.')
    parser.add_argument('file_path', type=str, help='Path to the Kraken output file')
    parser.add_argument('output_path', type=str, help='Path to output directory')
    
    parser.add_argument('--rows-per-chunk', type=int, default=20000, help='Rows per DataFrame chunk for pools')
    parser.add_argument('--max-bins-per-seq', type=int, default=None, help='Max bins per (seq_id, taxon) (default: None)')
    parser.add_argument('--shard-size', type=int, default=4096, help='Samples per shard (.pt)')
    parser.add_argument('--target-length', type=int, default=0,
                        help='Resample time to this length for shards (0 = pad to shard max)')
    parser.add_argument('--to-dtype', choices=['float32','float16','bfloat16'], default='float32',
                        help='Stored dtype for shard tensor')
    parser.add_argument('--min-tax-kmers', type=int, default=0,
                        help='Minimum k-mers assigned to a taxon for it to be considered')
    parser.add_argument('--threads', type=int, default=0, help='Number of worker processes (0=auto)')

    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    
    if args.threads == 1:
        globals_mod.NCBI = NCBITaxa() # Pre-initialize for single-threaded mode

    # Run extraction
    read_kraken_file(
        args.file_path, 
        args.output_path,
        rows_per_chunk=args.rows_per_chunk, 
        threads=args.threads, 
        max_bins_per_seq=args.max_bins_per_seq,
        shard_size=args.shard_size,
        target_length=args.target_length, 
        to_dtype=args.to_dtype,
        min_tax_kmers=args.min_tax_kmers
    )
        
    # Cleanup ETE3 temp dirs
    for tmpdir in glob.glob("/tmp/perseus_ete3db_*"):
        tmpdir = Path(tmpdir)
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
            LOG.debug("Deleted temp dir: %s", tmpdir)

if __name__ == '__main__':
    main()

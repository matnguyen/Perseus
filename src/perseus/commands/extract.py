#!/usr/bin/env python3

import os

# Must happen before NumPy/Pandas load BLAS.
os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1",
)
os.environ.setdefault(
    "OPENBLAS_NUM_THREADS",
    "1",
)
os.environ.setdefault(
    "MKL_NUM_THREADS",
    "1",
)
os.environ.setdefault(
    "NUMEXPR_NUM_THREADS",
    "1",
)

import re
import glob
import json
import shutil
import logging
import argparse as ap
import multiprocessing as mp

from pathlib import Path

import pandas as pd
from alive_progress import alive_bar

import perseus.utils.globals as globals_mod

from perseus.utils.constants import (
    CANONICAL_RANKS,
    N_CHANNELS,
)

from perseus.utils.tax_utils import (
    get_ncbi,
    normalize_taxid,
    canonicalize_rank,
    get_canonical_taxid_for_rank,
)

from perseus.features.init import (
    init_worker,
    init_feature_worker,
    effective_nprocs,
)

from perseus.features.tax_precompute import (
    init_taxonomy_map_worker,
    fetch_feature_maps,
)

from perseus.features.processing import (
    process_chunk_and_write,
    process_chunk_and_write_wrapper,
)

LOG = logging.getLogger(
    __name__
)

TOKEN_RE = re.compile(
    r"(\d+):(\d+)"
)


def collect_unique_taxids(
    file_path,
):
    """
    Lightweight first pass.

    Does NOT build:
        sequence -> taxid -> count

    It only collects the distinct taxids needed for
    taxonomy precomputation.
    """

    taxids = set()

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as fh:

        for line in fh:

            parts = (
                line.rstrip("\n")
                .split("\t", 4)
            )

            if len(parts) < 5:
                continue

            kmers = parts[4]

            for match in TOKEN_RE.finditer(
                kmers
            ):

                try:
                    tid = normalize_taxid(
                        int(
                            match.group(1)
                        )
                    )

                except Exception:
                    continue

                if tid is not None:
                    taxids.add(
                        int(tid)
                    )

    return taxids


def read_kraken_file(
    file_path,
    output_path,
    db_path,
    rows_per_chunk=5000,
    threads=0,
    max_bins_per_seq=None,
    shard_size=4096,
    target_length=1024,
    to_dtype="float32",
    min_tax_kmers=10,
):

    LOG.info(
        "Starting feature extraction..."
    )

    mess_true_file = None
    mess_input_file = None
    topk_taxa = None
    neg_extra = None
    is_training = False

    nprocs = (
        effective_nprocs()
        if threads == 0
        else int(threads)
    )

    # ========================================================
    # Lightweight unique-taxid pass
    # ========================================================

    LOG.info(
        "Collecting unique taxids..."
    )

    all_taxids = (
        collect_unique_taxids(
            file_path
        )
    )

    if not all_taxids:
        raise RuntimeError(
            "No valid taxonomic evidence "
            "was found"
        )

    LOG.info(
        "Found %d unique taxids",
        len(all_taxids),
    )

    # ========================================================
    # Precompute only taxonomy information actually used
    # ========================================================

    lineage_map = {}
    canonical_map = {}
    rank_idx_map = {}

    # Kept only for compatibility with existing init_worker.
    descendant_map = {}

    if nprocs <= 1:

        globals_mod.NCBI = get_ncbi(
            db_path
        )

        iterator = map(
            _fetch_feature_maps,
            all_taxids,
        )

        for (
            tid,
            lineage,
            canonicals,
            rank_idx,
        ) in iterator:

            lineage_map[tid] = (
                lineage
            )

            canonical_map[tid] = (
                canonicals
            )

            rank_idx_map[tid] = (
                rank_idx
            )

    else:

        # Taxonomy operations vary in cost.
        # 128 gives much better load balancing than using
        # rows_per_chunk (which may be ~20,000).
        map_chunksize = 128

        with mp.Pool(
            processes=nprocs,
            initializer=init_taxonomy_map_worker,
            initargs=(db_path,),
        ) as pool:

            iterator = pool.imap_unordered(
                fetch_feature_maps,
                all_taxids,
                chunksize=map_chunksize,
            )

            with alive_bar(
                len(all_taxids),
                title="Precomputing taxonomy",
            ) as bar:

                for (
                    tid,
                    lineage,
                    canonicals,
                    rank_idx,
                ) in iterator:

                    lineage_map[tid] = lineage
                    canonical_map[tid] = canonicals
                    rank_idx_map[tid] = rank_idx

                    bar()

    LOG.info(
        "Taxonomy precomputation complete"
    )

    # ========================================================
    # Output
    # ========================================================

    out_dir = Path(
        output_path
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Existing init_worker accepts tax_context.
    # Feature extraction no longer needs the huge map.
    tax_context = {}

    manager = mp.Manager()

    manifest_paths = (
        manager.list()
    )

    # ========================================================
    # Single actual processing pass
    # ========================================================

    with pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=[
            "Classified",
            "ID",
            "Taxonomy",
            "Length",
            "Kmers",
        ],
        dtype={
            "Classified": "category",
            "ID": "string",
            "Taxonomy": "string",
            "Length": "int32",
            "Kmers": "string",
        },
        engine="c",
        chunksize=rows_per_chunk,
    ) as reader:

        if nprocs <= 1:

            init_worker(
                tax_context,
                lineage_map,
                descendant_map,
                canonical_map,
                str(out_dir),
                db_path,
                shard_size,
                target_length,
                to_dtype,
                manifest_paths,
            )

            globals_mod._shared_rank_idx_map = (
                rank_idx_map
            )

            with alive_bar(
                title="Processing chunks",
                unknown="dots_waves",
            ) as bar:

                for chunk in reader:

                    process_chunk_and_write(
                        chunk,
                        max_bins_per_seq=(
                            max_bins_per_seq
                        ),
                        mess_true_file=(
                            mess_true_file
                        ),
                        mess_input_file=(
                            mess_input_file
                        ),
                        topk_taxa=topk_taxa,
                        min_tax_kmers=(
                            min_tax_kmers
                        ),
                        neg_extra=neg_extra,
                        is_training=(
                            is_training
                        ),
                    )

                    bar()

        else:

            with mp.Pool(
                processes=nprocs,
                initializer=init_feature_worker,
                initargs=(
                    tax_context,
                    lineage_map,
                    descendant_map,
                    canonical_map,
                    rank_idx_map,
                    str(out_dir),
                    db_path,
                    shard_size,
                    target_length,
                    to_dtype,
                    manifest_paths,
                ),
            ) as pool:

                results = pool.imap_unordered(
                    process_chunk_and_write_wrapper,
                    (
                        (
                            chunk,
                            max_bins_per_seq,
                            mess_true_file,
                            mess_input_file,
                            topk_taxa,
                            min_tax_kmers,
                            neg_extra,
                            is_training,
                        )
                        for chunk in reader
                    ),
                    chunksize=1,
                )

                with alive_bar(
                    title=(
                        "Processing chunks"
                    ),
                    unknown="dots_waves",
                ) as bar:

                    for _ in results:
                        # No gc.collect() after every chunk.
                        bar()

    # ========================================================
    # Manifest
    # ========================================================

    mani = {
        "source": str(
            file_path
        ),
        "outputs": list(
            manifest_paths
        ),
        "channels": N_CHANNELS,
        "target_length": int(
            target_length
        ),
        "dtype": str(
            to_dtype
        ),
        "shard_size": shard_size,
        "topk_taxa": topk_taxa,
        "min_tax_kmers": (
            min_tax_kmers
        ),
        "neg_extra": neg_extra,
        "labels": {
            "labels_per_rank": (
                f"length "
                f"{len(CANONICAL_RANKS)}; "
                f"equality per canonical rank"
            ),
            "rank_index": (
                f"index in CANONICAL_RANKS: "
                f"{CANONICAL_RANKS}"
            ),
        },
    }

    mani_path = (
        out_dir
        /
        "manifest.json"
    )

    with open(
        mani_path,
        "w",
    ) as fh:

        json.dump(
            mani,
            fh,
            indent=2,
        )

    LOG.info(
        "Wrote shard manifest -> %s",
        mani_path,
    )

    manager.shutdown()

def main():
    parser = ap.ArgumentParser(description='Chunked and parallel processing of Kraken output')
    parser.add_argument('file_path', type=str, help='Path to the Kraken output file')
    parser.add_argument('output_path', type=str, help='Path to output directory')
    parser.add_argument('db_dir', type=str, help="Directory containing ETE3 taxonomy database ")
    
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
    
    if not os.path.exists(args.db_dir):
        LOG.error("ETE3 taxonomy database not found at %s", args.db_dir / "taxa.sqlite")
        LOG.error("Run `perseus setup --db-dir %s` first", args.db_dir)
        raise SystemExit(1) 
    
    globals_mod.NCBI = get_ncbi(args.db_dir)  # Initialize NCBI in main process for single-threaded mode

    # Run extraction
    read_kraken_file(
        args.file_path, 
        args.output_path,
        args.db_dir,
        rows_per_chunk=args.rows_per_chunk, 
        threads=args.threads, 
        max_bins_per_seq=args.max_bins_per_seq,
        shard_size=args.shard_size,
        target_length=args.target_length, 
        to_dtype=args.to_dtype,
        min_tax_kmers=args.min_tax_kmers,
    )
        
    # Cleanup ETE3 temp dirs
    for tmpdir in glob.glob("/tmp/perseus_ete3db_*"):
        tmpdir = Path(tmpdir)
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
            LOG.debug("Deleted temp dir: %s", tmpdir)

if __name__ == '__main__':
    main()

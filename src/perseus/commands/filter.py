import os
import argparse
import logging
import torch
import glob
import pandas as pd
import numpy as np
from alive_progress import alive_bar
from pathlib import Path

from perseus.utils.tax_utils import get_ncbi
from perseus.utils.constants import CANONICAL_RANKS
from perseus.data.dataset import build_loader
from perseus.utils.filter_utils import select_one_row_per_seq
from perseus.models.initialize import (
    make_model,
    load_model,
    load_default_model
)

LOG = logging.getLogger(__name__)

def get_rank(ncbi, taxid):
    try:
        rank = ncbi.get_rank([taxid])[taxid]
    except KeyError:
        rank = 'no_rank'
    return rank

def get_lineage(ncbi, taxid):
    try:
        lineage = ncbi.get_lineage(taxid)
    except ValueError:
        lineage = []
    return lineage or []

def run_filter(args):
    LOG.info("Starting filter process...")
    LOG.info("Input shards directory: %s", args.input_shards)
    LOG.info("Input Kraken file: %s", args.input_kraken)
    LOG.info("Output path: %s", args.output_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("Using device: %s", device)
    
    LOG.debug("torch.cuda.is_available(): %s", torch.cuda.is_available())
    if device.type == "cuda":
        LOG.debug("CUDA device count: %d", torch.cuda.device_count())
        LOG.debug("CUDA device name: %s", torch.cuda.get_device_name(device))
    
    # Load model
    out_dim = len(CANONICAL_RANKS)
    if args.model_path is None:
        LOG.info("Using default model")
        model = load_default_model(out_dim, device=device)
    else:
        LOG.info("Loading model from: %s", args.model_path)
        model = make_model(out_dim, device)
        model = load_model(model, args.model_path, device)
    LOG.info("Model output dimension: %d", out_dim)
    model.eval()
    LOG.info("Model loaded successfully")
    
    # Build data loader
    LOG.info("Building data loader...")
    if not os.path.isdir(args.input_shards):
        LOG.error("Input shards path is not a directory: %s", args.input_shards)
        raise SystemExit(1)
    
    manifests = glob.glob(os.path.join(args.input_shards, "*manifest*.json"))
    LOG.debug("Manifest candidates: %s", manifests)
    if len(manifests) > 1:
        LOG.warning("Multiple manifest files found; using first: %s", manifests[0])
        
    if not manifests:
        LOG.error("No manifest files found in input directory: %s", args.input_shards)
        raise SystemExit(1)
    
    manifest_path = manifests[0]
    LOG.info("Using manifest: %s", manifest_path)
    _, data_loader = build_loader(args, manifest_path, args.batch_size, False, False, rank_filter=None)
    if len(data_loader) == 0:
        LOG.error("Data loader produced zero batches. No sequences will be scored")
        raise SystemExit(1)
    LOG.info("Data loader built successfully")
    LOG.info("Number of batches: %d", len(data_loader))
    LOG.info("Batch size: %d", args.batch_size)

    rows = []

    with torch.no_grad():
        LOG.info("Collecting model scores...")
        with alive_bar(len(data_loader), title="Scoring sequences") as bar:
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx == 0:
                    LOG.debug("First batch x shape: %s", tuple(batch["x"].shape))
                    LOG.debug("First batch mask shape: %s", tuple(batch["mask"].shape))
                    LOG.debug("First batch lengths shape: %s", tuple(batch["lengths"].shape))

                x = batch["x"].to(device, non_blocking=True).float()
                
                if batch_idx == 0:
                    LOG.debug("Input dtype after cast: %s", x.dtype)
                    
                mask = batch["mask"].to(device, non_blocking=True)
                extra = torch.log1p(batch["lengths"].to(device, non_blocking=True).float()).unsqueeze(1)
                logits = model(x, mask=mask, extra=extra)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                
                if batch_idx == 0:
                    LOG.debug("First batch logits shape: %s", tuple(logits.shape))
                    LOG.debug("First batch probs shape: %s", tuple(probs.shape))
                
                for i in range(len(probs)):
                    rows.append({
                        "sequence_id": batch["seq_id"][i],
                        "perseus_taxid": batch["taxon"][i],
                        "probs_per_rank": probs[i].tolist()
                    })
            
                bar()
                
    LOG.info("Collected scores for %d sequences", len(rows))

    # ============================================================
    # 1. BUILD SCORE DATAFRAME EFFICIENTLY
    # ============================================================
    #
    # Avoid:
    #   pd.DataFrame(rows)
    #   followed by one .apply() per probability column.
    #
    # Instead, convert all probability vectors into one contiguous
    # NumPy array and assign the columns directly.
    #

    LOG.info("Building score dataframe...")

    n_rows = len(rows)
    prob_cols = [f"prob_{rank}" for rank in CANONICAL_RANKS]

    sequence_ids = [r["sequence_id"] for r in rows]

    # int(...) handles Python ints, NumPy ints, and scalar torch tensors.
    perseus_taxids = np.fromiter(
        (int(r["perseus_taxid"]) for r in rows),
        dtype=np.int64,
        count=n_rows,
    )

    probs_array = np.asarray(
        [r["probs_per_rank"] for r in rows],
        dtype=np.float32,
    )

    if probs_array.ndim != 2 or probs_array.shape[1] != len(CANONICAL_RANKS):
        raise RuntimeError(
            f"Unexpected probability array shape: {probs_array.shape}; "
            f"expected (*, {len(CANONICAL_RANKS)})"
        )

    output_df = pd.DataFrame({
        "sequence_id": sequence_ids,
        "perseus_taxid": perseus_taxids,
    })

    # Assign all probability columns in one operation.
    output_df[prob_cols] = probs_array

    # Make sure merge key has the same dtype on both sides.
    output_df["sequence_id"] = output_df["sequence_id"].astype("string")

    # rows/probability Python objects are no longer needed.
    del rows
    del probs_array
    del sequence_ids
    del perseus_taxids

    LOG.info("Score dataframe built")


    # ============================================================
    # 2. LOAD ONLY THE KRAKEN COLUMNS WE ACTUALLY NEED
    # ============================================================
    #
    # The k-mer column can be enormous.
    #
    # Do NOT load it just to immediately drop it.
    #

    LOG.info("Loading Kraken output...")

    kraken_column_names = [
        "classified",
        "sequence_id",
        "kraken_taxonomy",
        "length",
        "kmers",
    ]

    kraken_df = pd.read_csv(
        args.input_kraken,
        sep="\t",
        header=None,
        names=kraken_column_names,

        # Important: completely skip the k-mer column.
        usecols=[
            "classified",
            "sequence_id",
            "kraken_taxonomy",
            "length",
        ],

        dtype={
            "classified": "string",
            "sequence_id": "string",
            "kraken_taxonomy": "string",
        },
    )

    # Extract final numeric taxid.
    #
    # This replaces:
    #
    #   .astype(str)
    #   .str.split()
    #   .str[-1]
    #   .str.strip(")")
    #
    # with a single extraction operation.
    taxid_text = kraken_df["kraken_taxonomy"].str.extract(
        r"(\d+)\)?\s*$",
        expand=False,
    )

    kraken_df["kraken_taxid"] = pd.to_numeric(
        taxid_text,
        errors="coerce",
    )

    bad_taxids = kraken_df["kraken_taxid"].isna()

    if bad_taxids.any():
        LOG.warning(
            "Failed to parse Kraken taxid for %d rows",
            int(bad_taxids.sum()),
        )

    # Invalid/missing taxonomy becomes taxid 0.
    kraken_df["kraken_taxid"] = (
        kraken_df["kraken_taxid"]
        .fillna(0)
        .astype(np.int64)
    )

    LOG.info(
        "Loaded Kraken output with %d entries",
        len(kraken_df),
    )


    # ============================================================
    # 3. MERGE SCORES WITH KRAKEN
    # ============================================================
    #
    # A LEFT join is normally sufficient here because Kraken is the
    # authoritative set of sequences we are filtering.
    #
    # This is cheaper than OUTER and avoids score-only rows.
    #
    # If you intentionally need rows present in scores but absent
    # from Kraken, change this back to how="outer".
    #

    LOG.info("Merging Kraken and Perseus outputs...")

    merged_df = kraken_df.merge(
        output_df,
        on="sequence_id",
        how="left",
        sort=False,
    )

    del output_df
    del kraken_df

    if merged_df.empty:
        LOG.error(
            "Merged dataframe is empty. "
            "No rows matched between Kraken and Perseus outputs"
        )
        raise SystemExit(1)

    LOG.info("Merged rows: %d", len(merged_df))

    # Any sequence that was not scored gets taxid 0.
    merged_df["perseus_taxid"] = (
        merged_df["perseus_taxid"]
        .fillna(0)
        .astype(np.int64)
    )


    # ============================================================
    # 4. OPTIONAL FULL OUTPUT
    # ============================================================

    if args.output_all:
        base, ext = os.path.splitext(args.output_path)
        full_output_path = f"{base}.full{ext}"

        merged_df.to_csv(
            full_output_path,
            sep="\t",
            index=False,
            float_format="%.6f",
        )

        LOG.info(
            "Full filtered Kraken output saved to %s",
            full_output_path,
        )


    # ============================================================
    # 5. BUILD ONE SHARED TAXONOMY LINEAGE CACHE
    # ============================================================
    #
    # Previously:
    #
    #   Kraken lineages
    #   +
    #   Perseus lineages
    #
    # were queried independently.
    #
    # That means overlapping taxids could hit ETE3 twice.
    #
    # Now every unique taxid gets exactly one lineage lookup.
    #

    ncbi = get_ncbi(args.db_dir)

    LOG.info(
        "Loaded ETE3 taxonomy database from %s",
        Path(args.db_dir).expanduser().resolve(),
    )

    unique_kraken = set(
        int(x)
        for x in merged_df["kraken_taxid"].unique()
        if int(x) > 0
    )

    unique_perseus = set(
        int(x)
        for x in merged_df["perseus_taxid"].unique()
        if int(x) > 0
    )

    all_taxids = unique_kraken | unique_perseus

    LOG.info(
        "Caching lineages for %d unique taxids "
        "(%d Kraken, %d Perseus)",
        len(all_taxids),
        len(unique_kraken),
        len(unique_perseus),
    )

    lineage_list_cache = {}

    with alive_bar(
        len(all_taxids),
        title="Caching lineages",
    ) as bar:

        for tx in all_taxids:
            lineage_list_cache[tx] = get_lineage(ncbi, tx)
            bar()

    # Invalid/unclassified taxid.
    lineage_list_cache[0] = []

    # Set representation specifically for fast membership checks.
    lineage_set_cache = {
        tx: frozenset(lineage)
        for tx, lineage in lineage_list_cache.items()
    }


    # ============================================================
    # 6. BULK-RANK LOOKUP FOR ALL TAXONOMY NODES
    # ============================================================
    #
    # This is an important improvement over:
    #
    #   for tx:
    #       ncbi.get_rank([tx])
    #
    # and:
    #
    #   for lineage:
    #       ncbi.get_rank(lineage)
    #
    # Gather every ancestor that we will ever need and call
    # ETE3 get_rank() ONCE.
    #

    LOG.info("Collecting unique taxonomy ancestors...")

    all_rank_taxids = set(unique_perseus)

    for lineage in lineage_list_cache.values():
        all_rank_taxids.update(lineage)

    LOG.info(
        "Bulk-querying ranks for %d taxonomy nodes",
        len(all_rank_taxids),
    )

    if all_rank_taxids:
        all_rank_map = ncbi.get_rank(
            list(all_rank_taxids)
        )
    else:
        all_rank_map = {}

    # Direct predicted-rank cache.
    rank_cache = {
        tx: all_rank_map.get(tx, "no_rank")
        for tx in unique_perseus
    }

    rank_cache[0] = "no_rank"


    # ============================================================
    # 7. PRECOMPUTE ANCESTOR AT EACH CANONICAL RANK
    # ============================================================

    canonical_rank_set = set(CANONICAL_RANKS)

    ancestor_at_rank_cache = {}

    LOG.info("Building rank-specific ancestor cache...")

    with alive_bar(
        len(unique_perseus),
        title="Caching ancestors at ranks",
    ) as bar:

        for tx in unique_perseus:

            lineage = lineage_list_cache.get(tx, [])

            rank_to_taxid = {}

            # deepest -> root
            for anc in reversed(lineage):

                rank = all_rank_map.get(anc)

                if rank == "kingdom":
                    rank = "superkingdom"

                if (
                    rank in canonical_rank_set
                    and rank not in rank_to_taxid
                ):
                    rank_to_taxid[rank] = anc

            ancestor_at_rank_cache[tx] = rank_to_taxid

            bar()

    ancestor_at_rank_cache[0] = {}

    LOG.info(
        "Cached %d unique lineages",
        len(lineage_list_cache),
    )

    LOG.info(
        "Cached %d predicted ranks",
        len(rank_cache),
    )

    LOG.info(
        "Cached %d ancestor maps",
        len(ancestor_at_rank_cache),
    )


    # ============================================================
    # 8. COMPUTE LINEAGE MEMBERSHIP WITHOUT iterrows()
    # ============================================================
    #
    # iterrows() is especially expensive on millions of rows because
    # pandas constructs a Series for every single row.
    #
    # Using NumPy arrays + set membership avoids that overhead.
    #

    LOG.info("Computing lineage membership...")

    kraken_taxids_array = merged_df[
        "kraken_taxid"
    ].to_numpy(
        dtype=np.int64,
        copy=False,
    )

    perseus_taxids_array = merged_df[
        "perseus_taxid"
    ].to_numpy(
        dtype=np.int64,
        copy=False,
    )

    empty_lineage = frozenset()

    merged_df["perseus_in_lineage"] = np.fromiter(
        (
            ptx in lineage_set_cache.get(
                ktx,
                empty_lineage,
            )
            for ktx, ptx in zip(
                kraken_taxids_array,
                perseus_taxids_array,
            )
        ),
        dtype=np.bool_,
        count=len(merged_df),
    )

    # Completely vectorized rank assignment.
    merged_df["perseus_predicted_rank"] = (
        merged_df["perseus_taxid"]
        .map(rank_cache)
        .fillna("no_rank")
    )

    del kraken_taxids_array
    del perseus_taxids_array


    # ============================================================
    # 9. SELECT FINAL CANDIDATE
    # ============================================================

    LOG.info("Selecting one candidate row per sequence...")

    filtered_df = select_one_row_per_seq(
        merged_df,
        sequence_col="sequence_id",
        ranks=[
            "superkingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ],
        thresholds=0.5,
        prefer_lineage=False,
        tie_breaker="sum_to_rank",
    )

    # merged_df can be large; release it as soon as possible.
    del merged_df


    # ============================================================
    # 10. FINAL TAXID BACKOFF WITHOUT apply(axis=1)
    # ============================================================
    #
    # Previously:
    #
    # filtered_df.apply(get_final_taxid_from_cache, axis=1)
    #
    # generated a pandas Series for every row.
    #
    # Instead, construct one taxid lookup per rank and use pandas.map().
    #

    LOG.info("Applying rank-specific taxonomic backoff...")

    ancestor_lookup_by_rank = {
        rank: {
            tx: rank_map.get(rank, tx)
            for tx, rank_map in ancestor_at_rank_cache.items()
        }
        for rank in CANONICAL_RANKS
    }

    base_taxids = (
        pd.to_numeric(
            filtered_df["perseus_taxid"],
            errors="coerce",
        )
        .astype("Int64")
    )

    chosen_ranks = filtered_df["chosen_rank"]

    final_taxids = base_taxids.copy()

    # Preserve old behavior: missing rank -> missing final taxid.
    final_taxids.loc[chosen_ranks.isna()] = pd.NA

    for rank in CANONICAL_RANKS:

        mask = (
            chosen_ranks.eq(rank)
            & base_taxids.notna()
        )

        if not mask.any():
            continue

        mapped = base_taxids.loc[mask].map(
            ancestor_lookup_by_rank[rank]
        )

        # Same fallback behavior as the old function:
        # if no rank-specific ancestor exists, retain base taxid.
        final_taxids.loc[mask] = (
            mapped
            .fillna(base_taxids.loc[mask])
            .astype("Int64")
        )

    filtered_df["perseus_taxid"] = final_taxids

    del ancestor_lookup_by_rank
    del final_taxids
    del base_taxids


    # ============================================================
    # 11. TAXID -> NAME IN ONE BULK QUERY
    # ============================================================

    final_unique_taxids = (
        filtered_df["perseus_taxid"]
        .dropna()
        .astype(np.int64)
        .unique()
        .tolist()
    )

    if final_unique_taxids:
        name_cache = ncbi.get_taxid_translator(
            final_unique_taxids
        )
    else:
        name_cache = {}

    filtered_df["perseus_taxonomy"] = (
        filtered_df["perseus_taxid"]
        .map(name_cache)
    )

    LOG.info(
        "Selected %d final rows",
        len(filtered_df),
    )


    # ============================================================
    # 12. DROP TEMPORARY COLUMNS
    # ============================================================

    filtered_df.drop(
        columns=[
            "perseus_in_lineage",
            "perseus_predicted_rank",
            "chosen_rank_ix",
        ],
        inplace=True,
        errors="ignore",
    )


    # ============================================================
    # 13. FINAL COLUMN ORDER
    # ============================================================

    ordered_cols = [
        "classified",
        "sequence_id",
        "kraken_taxonomy",
        "length",
        "kraken_taxid",
        "perseus_taxid",
        "perseus_taxonomy",
        "chosen_rank",
        "chosen_prob_at_rank",
    ] + prob_cols

    filtered_df = filtered_df[
        [
            col
            for col in ordered_cols
            if col in filtered_df.columns
        ]
    ]


    # ============================================================
    # 14. WRITE OUTPUT
    # ============================================================

    filtered_df.to_csv(
        args.output_path,
        sep="\t",
        index=False,
        float_format="%.6f",
    )

    LOG.info(
        "Filtered output saved to %s",
        args.output_path,
    )

    return filtered_df

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    
    parser = argparse.ArgumentParser(description="Filter Kraken outputs using a trained perseus model.")
    parser.add_argument('input_shards', type=str, 
                        help="Path to directory containing shard files; will search for 'manifest.json' manifest file.")
    parser.add_argument('input_kraken', type=str, 
                        help="Path to the Kraken output file to be filtered.")
    parser.add_argument('output_path', type=str, 
                        help="Path to save the filtered Kraken output.")
    parser.add_argument('db_dir', type=str, 
                    help="Directory containing ETE3 taxonomy database ")
    parser.add_argument('--batch-size', type=int, default=128,
                        help="Batch size for processing sequences.")
    parser.add_argument('--cache-shards', type=int, default=1, help="Shards kept in RAM per worker")
    parser.add_argument('--downcast', choices=["none","fp16"], default="fp16", help="Downcast shard tensors in cache")
    parser.add_argument('--cpu-float32', action="store_true", help="Cast samples to float32 on CPU before batching")
    parser.add_argument('--num-workers', type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument('--split-dir', type=str, default=None, 
                        help='Directory containing train/val splits (if applicable)')
    parser.add_argument('--seed', type=int, default=667, help="Random seed for reproducibility")
    parser.add_argument('--output-all', action="store_true", 
                        help="Output all model probabilities for each rank instead of just the predicted taxid.")
    parser.add_argument('--model-path', type=str,
                        help="Path to the trained perseus model file.")
    
    args = parser.parse_args()
    
    run_filter(args)


if __name__ == "__main__":
    main()
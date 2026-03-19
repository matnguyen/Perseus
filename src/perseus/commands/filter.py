import os
import argparse
import logging
import torch
import glob
import pandas as pd
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
    return lineage

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
                
    # Load Kraken output
    kraken_df = pd.read_csv(args.input_kraken, sep="\t", header=None, 
                            names=["classified", "sequence_id", "kraken_taxonomy", "length", "kmers"])
    kraken_df["kraken_taxid"] = (
        kraken_df["kraken_taxonomy"]
        .str.split()
        .str[-1]
        .str.strip(')')
        .astype(int)
    )
    if kraken_df["kraken_taxid"].isna().any():
        LOG.warning("Failed to parse some Kraken taxids")
    kraken_df.drop(columns=["kmers"], inplace=True)
    LOG.info("Loaded Kraken output with %d entries", len(kraken_df))
    
    output_df = pd.DataFrame(rows)
    merged_df = pd.merge(kraken_df, output_df, on=["sequence_id"], how="outer")
    if merged_df.empty:
        LOG.error("Merged dataframe is empty. No rows matched between Kraken and Perseus outputs")
        raise SystemExit(1)
    LOG.info("Kraken rows: %d", len(kraken_df))
    LOG.info("Scored rows: %d", len(rows))
    LOG.info("Merged rows: %d", len(merged_df))
    
    for idx, rank in enumerate(CANONICAL_RANKS):
        merged_df[f"prob_{rank}"] = merged_df["probs_per_rank"].apply(lambda x: x[idx] if isinstance(x, list) and len(x) > idx else None)
    
    merged_df.drop(columns=["probs_per_rank"], inplace=True)
    if args.output_all:
        base, ext = os.path.splitext(args.output_path)
        full_output_path = f"{base}.full{ext}"
        merged_df.to_csv(full_output_path, sep="\t", index=False, float_format="%.6f")
        LOG.info("Full filtered Kraken output saved to %s", full_output_path)
    
    merged_df['perseus_taxid'] = merged_df['perseus_taxid'].fillna(0)
    merged_df['perseus_taxid'] = merged_df['perseus_taxid'].astype(int)
    
    unique_truth = merged_df["kraken_taxid"].unique()
    
    ncbi = get_ncbi(args.db_dir)
    LOG.info("Loaded ETE3 taxonomy database from %s", Path(args.db_dir).expanduser().resolve())
    
    lineage_cache = {}

    with alive_bar(len(unique_truth), title="Caching lineages") as bar:
        for t in unique_truth:
            lineage_cache[t] = set(get_lineage(ncbi, t)) 
            bar()
            
    unique_perseus = merged_df["perseus_taxid"].unique()
    rank_cache = {}

    with alive_bar(len(unique_perseus), title="Caching ranks") as bar:
        for tx in unique_perseus:
            rank_cache[tx] = get_rank(ncbi, tx)
            bar()
            
    perseus_lineage_list_cache = {}

    with alive_bar(len(unique_perseus), title="Caching Perseus lineages") as bar:
        for tx in unique_perseus:
            perseus_lineage_list_cache[tx] = get_lineage(ncbi, tx)
            bar()

    ancestor_at_rank_cache = {}

    with alive_bar(len(unique_perseus), title="Caching ancestors at ranks") as bar:
        for tx in unique_perseus:
            lineage = perseus_lineage_list_cache[tx]
            lineage_ranks = ncbi.get_rank(lineage) if lineage else {}

            rank_to_taxid = {}
            for anc in reversed(lineage):   # deepest -> root
                r = lineage_ranks.get(anc)
                if r == 'kingdom':
                    r = 'superkingdom'
                if r in CANONICAL_RANKS and r not in rank_to_taxid:
                    rank_to_taxid[r] = anc

            ancestor_at_rank_cache[tx] = rank_to_taxid
            bar()
    
    LOG.debug("Cached %d Kraken taxid lineages", len(lineage_cache))
    LOG.debug("Cached %d Perseus taxid ranks", len(rank_cache))
    LOG.debug("Cached %d Perseus taxid lineages", len(perseus_lineage_list_cache))
    LOG.debug("Cached %d Perseus taxid ancestor-at-rank mappings", len(ancestor_at_rank_cache))
    
    perseus_in_lineage = []
    perseus_predicted_rank = []

    with alive_bar(len(merged_df)) as bar:
        for _, row in merged_df.iterrows():
            lin = lineage_cache[row["kraken_taxid"]]
            perseus_in_lineage.append(row["perseus_taxid"] in lin)
            perseus_predicted_rank.append(rank_cache[row["perseus_taxid"]])
            bar()

    merged_df["perseus_in_lineage"] = perseus_in_lineage
    merged_df["perseus_predicted_rank"] = perseus_predicted_rank
            
    LOG.info("Selecting one candidate row per sequence...")
    filtered_df = select_one_row_per_seq(
        merged_df,
        sequence_col="sequence_id",
        ranks=["superkingdom","phylum","class","order","family","genus","species"],
        thresholds=0.5,          
        prefer_lineage=False,
        tie_breaker="sum_to_rank",
    )
    
    def get_final_taxid_from_cache(row):
        base_taxid = row["perseus_taxid"]
        chosen_rank = row["chosen_rank"]

        if pd.isna(base_taxid) or pd.isna(chosen_rank):
            return None

        try:
            base_taxid = int(base_taxid)
        except Exception:
            return None

        return ancestor_at_rank_cache.get(base_taxid, {}).get(chosen_rank, base_taxid)
    
    filtered_df["perseus_taxid"] = filtered_df.apply(get_final_taxid_from_cache, axis=1)
    
    final_taxids = pd.Series(filtered_df["perseus_taxid"].dropna().unique()).astype(int).tolist()
    name_cache = ncbi.get_taxid_translator(final_taxids) if final_taxids else {}
    filtered_df["perseus_taxonomy"] = filtered_df["perseus_taxid"].map(name_cache)
    
    LOG.info("Selected %d final rows", len(filtered_df))
    
    filtered_df.drop(
        columns=["perseus_in_lineage", "perseus_predicted_rank", "chosen_rank_ix"],
        inplace=True,
        errors="ignore",
    )

    prob_cols = [f"prob_{rank}" for rank in CANONICAL_RANKS]
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

    filtered_df = filtered_df[[c for c in ordered_cols if c in filtered_df.columns]]
    
    filtered_output_path = args.output_path
    filtered_df.to_csv(filtered_output_path, sep="\t", index=False, float_format="%.6f")
    LOG.info("Filtered output saved to %s", filtered_output_path)
    
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
import os
import argparse
import logging
import torch
import pickle
import pandas as pd
from alive_progress import alive_bar
from ete3 import NCBITaxa

from perseus.utils.constants import CANONICAL_RANKS
from perseus.data.dataset import build_loader
from perseus.utils.filter_utils import select_one_row_per_seq
from perseus.models.initialize import (
    make_model,
    load_model
)

ncbi = NCBITaxa()

def get_rank(taxid):
    try:
        rank = ncbi.get_rank([taxid])[taxid]
    except KeyError:
        rank = 'no_rank'
    return rank

def get_lineage(taxid):
    try:
        lineage = ncbi.get_lineage(taxid)
    except ValueError:
        lineage = []
    return lineage

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    parser = argparse.ArgumentParser(description="Filter Kraken outputs using a trained perseus model.")
    parser.add_argument("--input-shards", type=str, required=True,
                        help="Path to the input manifest file containing sequences to filter.")
    parser.add_argument("--input-kraken", type=str, required=True,
                        help="Path to the Kraken output file to be filtered.")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained perseus model file.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for processing sequences.")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to save the filtered Kraken output.")
    parser.add_argument("--cache-shards", type=int, default=1, help="Shards kept in RAM per worker")
    parser.add_argument("--downcast", choices=["none","fp16"], default="fp16", help="Downcast shard tensors in cache")
    parser.add_argument("--cpu-float32", action="store_true", help="Cast samples to float32 on CPU before batching")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--split-dir", type=str, default=None, 
                        help='Directory containing train/val splits (if applicable)')
    parser.add_argument("--seed", type=int, default=667, help="Random seed for reproducibility")
    parser.add_argument("--select-one-per-seq", action="store_true", 
                        help="Select one row per sequence ID from Kraken output using model probabilities")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load model
    out_dim = len(CANONICAL_RANKS)
    model = make_model(args, out_dim, device)
    model = load_model(model, args.model_path, device)
    logging.info("Model loaded successfully.")
    
    # Build data loader
    logging.info("Building data loader...")
    _, data_loader = build_loader(args, args.input_shards, args.batch_size, False, False, rank_filter=None)
    logging.info("Data loader built successfully.")

    rows = []

    with torch.no_grad():
        logging.info("Collecting model scores...")
        with alive_bar(len(data_loader), title="Scoring sequences") as bar:
            for batch in data_loader:
                # Forward 
                x = batch["x"].to(device, non_blocking=True).float()
                mask = batch["mask"].to(device, non_blocking=True)
                extra = torch.log1p(batch["lengths"].to(device, non_blocking=True).float()).unsqueeze(1)
                logits = model(x, mask=mask, extra=extra)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                
                for i in range(len(probs)):
                    rows.append({
                        "sequence_id": batch["seq_id"][i],
                        "perseus_taxid": batch["taxon"][i],
                        "probs_per_rank": probs[i].tolist()
                    })
            
                bar()
    # Load Kraken output
    kraken_df = pd.read_csv(args.input_kraken, sep="\t", header=None, 
                            names=["classified", "sequence_id", "kraken_taxonomy", "length", "kmers"])
    kraken_df["kraken_taxid"] = kraken_df["kraken_taxonomy"].str.split().str[-1].str.strip(')')
    kraken_df.drop(columns=["kmers"], inplace=True)
    logging.info(f"Loaded Kraken output with {len(kraken_df)} entries.")
    
    output_df = pd.DataFrame(rows)
    merged_df = pd.merge(kraken_df, output_df, on=["sequence_id"], how="outer")
    logging.info(f"Merged data has {len(merged_df)} entries.")
    
    for idx, rank in enumerate(CANONICAL_RANKS):
        merged_df[f"prob_{rank}"] = merged_df["probs_per_rank"].apply(lambda x: x[idx] if isinstance(x, list) and len(x) > idx else None)
    
    merged_df.drop(columns=["probs_per_rank"], inplace=True)
    merged_df.to_csv(args.output_path, sep="\t", index=False)
    logging.info(f"Filtered Kraken output saved to {args.output_path}.")
    
    if args.select_one_per_seq:
        merged_df['perseus_taxid'] = merged_df['perseus_taxid'].fillna(0)
        merged_df['perseus_taxid'] = merged_df['perseus_taxid'].astype(int)
        
        unique_truth = merged_df["kraken_taxid"].unique()
        lineage_cache = {}

        with alive_bar(len(unique_truth), title="Caching lineages", force_tty=True) as bar:
            for t in unique_truth:
                lineage_cache[t] = set(get_lineage(t)) 
                bar()
                
        unique_perseus = merged_df["perseus_taxid"].unique()
        rank_cache = {}

        with alive_bar(len(unique_perseus), title="Caching ranks", force_tty=True) as bar:
            for tx in unique_perseus:
                rank_cache[tx] = get_rank(tx)
                bar()
                
        perseus_in_lineage = []
        perseus_predicted_rank = []

        with alive_bar(len(merged_df), force_tty=True) as bar:
            for _, row in merged_df.iterrows():
                lin = lineage_cache[row["kraken_taxid"]]
                perseus_in_lineage.append(row["perseus_taxid"] in lin)
                perseus_predicted_rank.append(rank_cache[row["perseus_taxid"]])
                bar()

        merged_df["perseus_in_lineage"] = perseus_in_lineage
        merged_df["perseus_predicted_rank"] = perseus_predicted_rank
                
        logging.info("Selecting one row per sequence ID based on model probabilities...")
        filtered_df = select_one_row_per_seq(
            merged_df,
            sequence_col="sequence_id",
            ranks=["superkingdom","phylum","class","order","family","genus","species"],
            thresholds=0.5,          
            prefer_lineage=False,
            tie_breaker="sum_to_rank",
        )
        filtered_output_path = args.output_path.replace(".tsv", "_one_per_seq.tsv")
        filtered_df.to_csv(filtered_output_path, sep="\t", index=False)
        logging.info(f"Filtered one-per-sequence output saved to {filtered_output_path}.")
    
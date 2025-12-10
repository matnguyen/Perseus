import os
import argparse
import logging
import torch
import pickle
import pandas as pd
from alive_progress import alive_bar

from taxoncnn.utils.constants import CANONICAL_RANKS
from taxoncnn.data.dataset import build_loader
from taxoncnn.trainer.evaluate import _collect_scores_per_rank
from taxoncnn.models.initialize import (
    make_model,
    load_model
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    parser = argparse.ArgumentParser(description="Filter Kraken outputs using a trained TaxonCNN model.")
    parser.add_argument("--input-shards", type=str, required=True,
                        help="Path to the input manifest file containing sequences to filter.")
    parser.add_argument("--input-kraken", type=str, required=True,
                        help="Path to the Kraken output file to be filtered.")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained TaxonCNN model file.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for processing sequences.")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to save the filtered Kraken output.")
    parser.add_argument("--cache-shards", type=int, default=1, help="Shards kept in RAM per worker")
    parser.add_argument("--downcast", choices=["none","fp16"], default="fp16", help="Downcast shard tensors in cache")
    parser.add_argument("--cpu-float32", action="store_true", help="Cast samples to float32 on CPU before batching")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--calibration-dir", type=str, default=None, help="Directory containing calibrators")
    parser.add_argument("--model", type=str, default="cnn", help="Model architecture to use")
    
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
    _, data_loader = build_loader(args, args.input_shards, args.batch_size, False, rank_filter=None)
    logging.info("Data loader built successfully.")
    
    # Load calibrators if provided
    if args.calibration_dir is not None:
        logging.info("Loading calibrators...")
        calibrators = {}
        for r,name in enumerate(CANONICAL_RANKS):
            with open(os.path.join(args.calibration_dir, f"calibrator_{name}.pkl"), "rb") as f:
                calibrators[r] = pickle.load(f)

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
                        "taxon": batch["taxon"][i],
                        "probs_per_rank": probs[i].tolist()
                    })
                
                bar()
    # Load Kraken output
    kraken_df = pd.read_csv(args.input_kraken, sep="\t", header=None, 
                            names=["classified", "sequence_id", "kraken_taxonomy", "length", "kmers"])
    kraken_df["kraken_taxid"] = kraken_df["kraken_taxonomy"].str.split().str[-1].str.strip(')')
    logging.info(f"Loaded Kraken output with {len(kraken_df)} entries.")
    
    output_df = pd.DataFrame(rows)
    merged_df = pd.merge(kraken_df, output_df, on=["sequence_id"], how="outer")
    logging.info(f"Merged data has {len(merged_df)} entries.")
    
    for idx, rank in enumerate(CANONICAL_RANKS):
        merged_df[f"prob_{rank}"] = merged_df["probs_per_rank"].apply(lambda x: x[idx] if isinstance(x, list) and len(x) > idx else None)
    
    merged_df.drop(columns=["probs_per_rank"], inplace=True)
    merged_df.to_csv(args.output_path, sep="\t", index=False)
    logging.info(f"Filtered Kraken output saved to {args.output_path}.")
    
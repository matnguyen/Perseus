import os
import argparse
import logging
import torch
import pickle
import pandas as pd

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
    _, data_loader = build_loader(args, args.input_shards, args.batch, False, rank_filter=None)
    logging.info("Data loader built successfully.")
    
    # Load calibrators if provided
    if args.calibration_dir is not None:
        logging.info("Loading calibrators...")
        calibrators = {}
        for r,name in enumerate(CANONICAL_RANKS):
            with open(os.path.join(args.calibration_dir, f"calibrator_{name}.pkl"), "rb") as f:
                calibrators[r] = pickle.load(f)
    
    if args.calibration_dir is not None:
        logging.info("Collecting scores with calibration...")
        per_rank = _collect_scores_per_rank(model, data_loader, device, calibrators=calibrators, use_calibration=True)
    else:
        logging.info("Collecting scores without calibration...")
        per_rank = _collect_scores_per_rank(model, data_loader, device)
        
    # Load Kraken output
    kraken_df = pd.read_csv(args.input_kraken, sep="\t", header=None, 
                            names=["classified", "sequence_id", "taxonomy", "length", "kmers"])
    kraken_df.set_index("sequence_id", inplace=True)
    logging.info(f"Loaded Kraken output with {len(kraken_df)} entries.")
    
    import pdb
    pdb.set_trace()
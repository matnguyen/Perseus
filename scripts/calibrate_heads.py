#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from taxoncnn.utils.constants import CANONICAL_RANKS
from taxoncnn.data.dataset import build_loader
from taxoncnn.models.initialize import (
    make_model,
    load_model
)
from taxoncnn.trainer.calibration import (
    collect_head_outputs,
    fit_isotonic_per_head,
    save_calibrators,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    parser = argparse.ArgumentParser(description="Calibrate CNN1D_CF (single head) with isotonic regression")
    parser.add_argument("--shards", required=True,
                        help="Dir or manifest.json for calibration shards")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="Path to save calibrator .pkl")
    parser.add_argument("--label-key", default="y_rank", choices=["y_any", "y_rank"],
                        help="Which label to calibrate against")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-shards", type=int, default=1, help="Shards kept in RAM per worker")
    parser.add_argument("--downcast", choices=["none","fp16"], default="fp16", help="Downcast shard tensors in cache")
    parser.add_argument("--cpu-float32", action="store_true", help="Cast samples to float32 on CPU before batching")
    parser.add_argument("--model", choices=["cnn","restcn"], default="cnn")
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logging.info(f"[calibration] Using device: {device}")
    # --- Build calibration dataset + loader ---
    logging.info(f"[calibration] Building calibration loader from shards: {args.shards}")
    _, calib_loader   = build_loader(args, args.shards, args.batch_size, False, rank_filter=None)

    # --- Load model ---
    logging.info(f"[calibration] Loading model from checkpoint: {args.checkpoint}")
    out_dim = len(CANONICAL_RANKS)
    model = make_model(args, out_dim, device)
    model = load_model(model, args.checkpoint, device)  

    logging.info(f"[calibration] Collecting head outputs on calibration set")
    # --- Collect scores + labels ---
    scores, labels = collect_head_outputs(
        model=model,
        dataloader=calib_loader,
        device=device
    )

    logging.info(f"[calibration] Fitting isotonic regression per head") 
    # --- Fit calibrator ---
    iso = fit_isotonic_per_head(scores, labels)

    # --- Save ---
    save_calibrators(iso, Path(args.out))
    logging.info(f"[calibration] Saved calibrators to {args.out}")
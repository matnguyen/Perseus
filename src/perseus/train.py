#!/usr/bin/env python3
import os
import gc
import argparse
import logging
import torch
from pathlib import Path

import torch.multiprocessing as mp
try: mp.set_start_method("forkserver", force=True)
except RuntimeError: pass
try: mp.set_sharing_strategy("file_system")
except RuntimeError: pass

from perseus.trainer.train import train
from perseus.data.dataset import build_loader
from perseus.models.initialize import make_model
from perseus.utils.constants import (
    CANONICAL_RANKS,
    RANK_INDEX,
    N_CHANNELS,
    CROP_MAX_T
)

# -------------------------
# Logging
# -------------------------
LOG = logging.getLogger("train_by_rank")

def setup_logging(level: str = "INFO"):
    """
    Configure logging for the training script

    Args:
        level (str, optional): Logging level as a string (e.g., "INFO", "DEBUG"). Defaults to "INFO"

    Returns:
        None
    """
    level = level.upper()
    if level not in ("DEBUG","INFO","WARNING","ERROR","CRITICAL"):
        level = "INFO"
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

# Limit BLAS thread fan-out (helps RAM too)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


if __name__ == "__main__":
    """
    Entry point for training from (train, val) shard manifests with Option-B labels.

    Parses command-line arguments, sets up logging and device, resolves input paths,
    builds data loaders, instantiates the model, and runs training for the selected
    target mode and rank filtering options.
    """
    ap = argparse.ArgumentParser("Train from (train,val) shard manifests with Option-B labels (optimized I/O).")
    ap.add_argument("--train", required=False, help="Train shard directory OR train_manifest.json")
    ap.add_argument("--val",   required=False, help="Val shard directory OR val_manifest.json")
    ap.add_argument("--input", required=False, help="Shard directory OR single manifest (used for both if --val missing)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=2)  # <=2 to avoid N× memory
    ap.add_argument("--model", choices=["cnn","restcn"], default="cnn")
    ap.add_argument("--save",   default="model_cf.pt")

    ap.add_argument("--target", choices=["any","rank","per-rank"], default="any")

    group = ap.add_mutually_exclusive_group()
    group.add_argument("--rank", choices=CANONICAL_RANKS, help="Use only samples with this predicted canonical rank")
    group.add_argument("--ranks", action="store_true", help="Train one model per predicted canonical rank (loop)")

    ap.add_argument("--rank_cache", default=None, help="Optional cache path for rank index")
    ap.add_argument("--log-level", default="INFO", help="DEBUG | INFO | WARNING | ERROR | CRITICAL")
    ap.add_argument("--crop-max", type=int, default=CROP_MAX_T, help="Max crop length for TRAIN loader (no crop for VAL)")
    # Loader/Dataset memory knobs
    ap.add_argument("--cache-shards", type=int, default=1, help="Shards kept in RAM per worker")
    ap.add_argument("--downcast", choices=["none","fp16"], default="fp16", help="Downcast shard tensors in cache")
    ap.add_argument("--cpu-float32", action="store_true", help="Cast samples to float32 on CPU before batching")

    args = ap.parse_args()
    setup_logging(args.log_level)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("Using device: %s (CUDA avail=%s)", device, torch.cuda.is_available())

    # Resolve inputs
    if args.train or args.val:
        if not args.train or not args.val:
            raise SystemExit("Please provide both --train and --val (paths to shard dirs or manifests).")
        train_input = args.train
        val_input   = args.val
    else:
        if not args.input:
            raise SystemExit("Provide either --train and --val, or a single --input.")
        train_input = val_input = args.input
        LOG.warning("--train/--val not provided; using --input for both.")

    out_dim = 1 if args.target in ("any","rank") else len(CANONICAL_RANKS)

    if args.ranks:
        for rk in CANONICAL_RANKS:
            LOG.info("=== Predicted Rank: %s ===", rk)
            _, train_loader = build_loader(args, train_input, args.batch, True,  rank_filter=rk)
            _, val_loader   = build_loader(args, val_input,   args.batch, False, rank_filter=rk)
            model = make_model(args, out_dim, device)
            save_path = f"{Path(args.save).with_suffix('')}_{args.target}_{rk}.pt"
            rank_idx_gate = RANK_INDEX[rk] if args.target == "rank" else None
            train(model, train_loader, val_loader, device,
                  target_mode=args.target, rank_idx_for_gate=rank_idx_gate,
                  epochs=args.epochs, lr=args.lr, save_path=save_path)
            torch.cuda.empty_cache(); gc.collect()

    if args.rank:
        rk = args.rank
        LOG.info("Training single model for predicted rank='%s'", rk)
        _, train_loader = build_loader(args, train_input, args.batch, True,  rank_filter=rk)
        _, val_loader   = build_loader(args, val_input,   args.batch, False, rank_filter=rk)
        model = make_model(args, out_dim, device)
        save_path = f"{Path(args.save).with_suffix('')}_{args.target}_{rk}.pt"
        rank_idx_gate = RANK_INDEX[rk] if args.target == "rank" else None
        train(model, train_loader, val_loader, device,
              target_mode=args.target, rank_idx_for_gate=rank_idx_gate,
              epochs=args.epochs, lr=args.lr, save_path=save_path)

    LOG.info("Training on ALL samples (no rank filter). target=%s", args.target)
    _, train_loader = build_loader(args, train_input, args.batch, True,  rank_filter=None)
    _, val_loader   = build_loader(args, val_input,   args.batch, False, rank_filter=None)
    model = make_model(args, out_dim, device)
    save_path = f"{Path(args.save).with_suffix('')}_{args.target}.pt"
    train(model, train_loader, val_loader, device,
          target_mode=args.target, rank_idx_for_gate=None,
          epochs=args.epochs, lr=args.lr, save_path=save_path)

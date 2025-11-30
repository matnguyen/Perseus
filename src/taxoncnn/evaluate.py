import os
import argparse
import torch
import logging
import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle

from taxoncnn.utils.constants import CANONICAL_RANKS
from taxoncnn.data.dataset import build_loader
from taxoncnn.trainer.evaluate import _collect_scores_per_rank
from taxoncnn.models.initialize import (
    make_model,
    load_model
)
from taxoncnn.trainer.metrics import (
    binary_auroc,
    binary_aupr,
    precision_recall_curve_from_scores,
    confusion_matrix_from_threshold,
    f1_from_counts
)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    parser = argparse.ArgumentParser(description="Evaluate TaxonCNN model")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint")
    parser.add_argument("--shards", required=True, help="Path to data shards for evaluation")
    parser.add_argument("--model", choices=["cnn","restcn"], default="cnn")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--cache-shards", type=int, default=1, help="Shards kept in RAM per worker")
    parser.add_argument("--downcast", choices=["none","fp16"], default="fp16", help="Downcast shard tensors in cache")
    parser.add_argument("--cpu-float32", action="store_true", help="Cast samples to float32 on CPU before batching")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--calibration-dir", type=str, default=None, help="Directory containing calibrators")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[evaluation] Using device: {device}")
    
    # Load model
    out_dim = len(CANONICAL_RANKS)
    model = make_model(args, out_dim, device)
    model = load_model(model, args.checkpoint, device)  
    
    # Build evaluation dataset + loader
    logging.info(f"[evaluation] Building evaluation loader from shards: {args.shards}")
    _, eval_loader = build_loader(args, args.shards, args.batch, False, rank_filter=None)
    
    # Load calibrators if provided
    calibrators = {}
    for r,name in enumerate(CANONICAL_RANKS):
        with open(os.path.join(args.calibration_dir, f"calibrator_{name}.pkl"), "rb") as f:
            calibrators[r] = pickle.load(f)
    
    # Collect scores per rank
    logging.info(f"[evaluation] Collecting scores per rank on evaluation set")
    
    if args.calibration_dir is not None:
        per_rank = _collect_scores_per_rank(model, eval_loader, device, calibrators=calibrators, use_calibration=True)
    else:
        per_rank = _collect_scores_per_rank(model, eval_loader, device)
    
    # Output results
    rows = []
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    table_path = os.path.join(args.output_dir, "rank_metrics.tsv")
    summary_path = os.path.join(args.output_dir, "summary.txt")
    
    with open(table_path, "w", newline="") as tsvfile, open(summary_path, "w") as summaryfile:
        writer = csv.writer(tsvfile, delimiter="\t")
        writer.writerow(["Rank", "N", "AUROC", "AUPR", "thr(0.5)", "TP", "FP", "FN"])
        summaryfile.write("Rank\tN\tAUROC\tAUPR\tthr(0.5)\tTP\tFP\tFN\n")

        for r_ix, rk in enumerate(CANONICAL_RANKS):
            y_true, y_score = per_rank[r_ix]
            if y_true.size == 0:
                msg = f"[{rk}] No valid labeled samples; skipping plots."
                print(msg)
                summaryfile.write(msg + "\n")
                rows.append((rk, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
                writer.writerow([rk, 0, "nan", "nan", "nan", "nan", "nan", "nan"])
                continue

            auroc = binary_auroc(y_true, y_score)
            aupr  = binary_aupr(y_true, y_score)

            # Best-F1 threshold search
            thresholds = np.linspace(0.01, 0.99, 99)
            best = (0.0, 0.5, 0.0, 0.0)  # f1, thr, prec, rec
            for thr in thresholds:
                tp, fp, fn, tn = confusion_matrix_from_threshold(y_true, y_score, thr)
                f1, prec, rec = f1_from_counts(tp, fp, fn)
                if f1 > best[0]:
                    best = (f1, thr, prec, rec)
            f1_best, thr_best, prec_best, rec_best = best

            pr_path = os.path.join(args.output_dir, f"pr_curve_{rk}.png")
            cm_path = os.path.join(args.output_dir, f"confusion_{rk}.png")
            # Plot at default 0.5 for display; you can also use thr_best here if desired.
            tp, fp, fn, tn = confusion_matrix_from_threshold(y_true, y_score, 0.5)
            # PR curve plot
            prec, rec = precision_recall_curve_from_scores(y_true, y_score)
            plt.figure(figsize=(6,5))
            plt.plot(rec, prec, linewidth=2)
            plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.title(f"{rk} — PR (AUPR={aupr:.3f})")
            plt.grid(True, linestyle="--", linewidth=0.5)
            plt.tight_layout(); plt.savefig(pr_path, dpi=200); plt.close()
            # Confusion matrix plot
            cm = np.array([[tn, fp],[fn, tp]], dtype=np.int64)
            plt.figure(figsize=(5,4))
            plt.imshow(cm, aspect="equal")
            plt.xticks([0,1], ["Pred 0","Pred 1"]); plt.yticks([0,1], ["True 0","True 1"])
            for (i, j), v in np.ndenumerate(cm):
                plt.text(j, i, str(v), ha="center", va="center")
            plt.title(f"{rk} — Confusion @ thr=0.50")
            plt.xlabel("Predicted"); plt.ylabel("True")
            plt.tight_layout(); plt.savefig(cm_path, dpi=200); plt.close()

            rows.append((rk, y_true.size, auroc, aupr, 0.5, tp, fp, fn))
            msg = (f"[{rk}] N={y_true.size} | AUROC={auroc:.4f} | AUPR={aupr:.4f} | "
                f"BestF1 thr={thr_best:.2f} (F1={f1_best:.4f}, P={prec_best:.4f}, R={rec_best:.4f}) | "
                f"PR: {pr_path} | CM: {cm_path}")
            print(msg)
            summaryfile.write(msg + "\n")
            writer.writerow([rk, y_true.size, f"{auroc:.4f}", f"{aupr:.4f}", "0.50", tp, fp, fn])

        print("\nRank\tN\tAUROC\tAUPR\tthr(0.5)\tTP\tFP\tFN")
        summaryfile.write("\nRank\tN\tAUROC\tAUPR\tthr(0.5)\tTP\tFP\tFN\n")
        for row in rows:
            rk, N, auroc, aupr, thr, tp, fp, fn = row
            line = f"{rk}\t{N}\t{auroc:.4f}\t{aupr:.4f}\t{thr:.2f}\t{tp}\t{fp}\t{fn}"
            print(line)
            summaryfile.write(line + "\n")
import os
import argparse
import torch
import logging
import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd 

from ete3 import NCBITaxa
from alive_progress import alive_bar

from perseus.utils.constants import CANONICAL_RANKS
from perseus.data.dataset import build_loader
from perseus.trainer.evaluate import _collect_scores_per_rank
from perseus.models.initialize import (
    make_model,
    load_model
)
from perseus.trainer.metrics import (
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
    
    parser = argparse.ArgumentParser(description="Evaluate Perseus model")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint")
    parser.add_argument("--kraken-input", required=True, help="Path to Kraken input file for evaluation")
    parser.add_argument("--map-file", required=True, help="Path to file mapping anonymized reads to taxid for evaluation")
    parser.add_argument("--true-file", required=True, help="Path to true labels file for evaluation")
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
    if args.calibration_dir is not None:
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
            
    kraken_df = pd.read_csv(args.kraken_input, sep="\t", 
                            names=["classified", "seq_id", "taxonomy", "length", "kmers"], index_col=None).drop(columns=["kmers"])
    
    map_df = pd.read_csv(args.map_file, sep="\t", names=["read", "map"])
    map_df["accession"] = map_df["map"].str.split('/').str[0].str[:-1]
    
    true_df = pd.read_csv(args.true_file, sep="\t")
    
    if (true_df["tax_id"] == 0).all():
        true_df["tax_id"] = true_df["fasta"].str.split('__').str[2]
    true_df['fasta'] = true_df['fasta'].str.split('__').str[-1]
    
    reads_mapped = pd.merge(true_df, map_df, left_on='fasta', right_on='accession', how='inner')
    kraken_species_true = pd.merge(kraken_df, reads_mapped, left_on='seq', right_on='read', how='inner')
    
    kraken_species_true['taxid'] = kraken_species_true['taxonomy'].str.split().str[-1].str.strip(')')
    kraken_species_true = kraken_species_true[['classified', 'path', 'seq', 'tax_id', 'taxid']]
    kraken_species_true.columns = ['classified', 'path', 'seq', 'true_taxid', 'taxid']
    
    df = kraken_species_true.copy()
    
    # --- convenience booleans ---
    is_classified   = df["classified"].eq("C")
    is_unclassified = ~is_classified
    is_included     = df["path"].str.startswith("included")
    is_excluded     = df["path"].str.startswith("excluded")
    
    ncbi = NCBITaxa()
    _lineage_cache = {}
    def lineage_set(tid: int):
        tid = int(tid)
        if tid not in _lineage_cache:
            _lineage_cache[tid] = set(ncbi.get_lineage(tid))
        return _lineage_cache[tid]

    def pred_in_true_lineage(pred, truth) -> bool:
        try:
            return int(pred) in lineage_set(int(truth))
        except Exception:
            return False
        
    df["pred_in_true_lineage"] = False
    mask = is_classified & df["taxid"].notna()
    df.loc[mask, "pred_in_true_lineage"] = [
        pred_in_true_lineage(p, t) for p, t in zip(df.loc[mask, "taxid"], df.loc[mask, "true_taxid"])
    ]
    
    # --- assign TP/FP/FN/TN per your definitions ---
    # fp: classified & NOT in lineage of true_taxid
    # fn: unclassified & included
    # tp: classified & in lineage (includes exact match)
    # tn: unclassified & excluded
    conds = [
        is_classified & df["pred_in_true_lineage"],         # TP
        is_classified & ~df["pred_in_true_lineage"],        # FP
        is_unclassified & is_included,                      # FN
        is_unclassified & is_excluded,                      # TN
    ]
    choices = ["TP", "FP", "FN", "TN"]
    df["label"] = np.select(conds, choices, default="OTHER")
    
    TP = (df["label"] == "TP").sum()
    FP = (df["label"] == "FP").sum()
    FN = (df["label"] == "FN").sum()
    TN = (df["label"] == "TN").sum()
    
    # "Operational" (among classified) — what % of assignments were wrong?
    false_assignment_rate = FP / (TP + FP) if (TP + FP) else np.nan
    precision             = TP / (TP + FP) if (TP + FP) else np.nan
    recall_TPR            = TP / (TP + FN) if (TP + FN) else np.nan
    specificity_TNR       = TN / (TN + FP) if (TN + FP) else np.nan
    FPR_classical         = FP / (FP + TN) if (FP + TN) else np.nan
    
    summary_overall = {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "false_assignment_rate": false_assignment_rate,  # FP / (TP+FP) — the plot you likely want
        "precision": precision,
        "recall_TPR": recall_TPR,
        "specificity_TNR": specificity_TNR,
        "FPR_classical": FPR_classical,                  # requires unclassified rows to populate TN
    }
    
    R = len(CANONICAL_RANKS)
    
    all_preds = []
    seq_ids = []
    taxons = []
    per_rank_scores = [[] for _ in range(R)]
    
    row = []
    
    with torch.no_grad():
        with alive_bar(len(eval_loader)) as bar:
            for batch in eval_loader:
                x     = batch["x"].to(device, non_blocking=True).float()
                msk   = batch["mask"].to(device, non_blocking=True)
                extra = torch.log1p(batch["lengths"].to(device).float()).unsqueeze(1)
                logits = model(x, mask=msk, extra=extra)                       
                probs  = torch.sigmoid(logits).detach().cpu().numpy()  
                
                for i in range(len(probs)):
                    row.append({
                    "seq_id": batch["bundle"][i]["seq_id"],
                    "taxon": batch["bundle"][i]["taxon"],
                    "preds_per_rank": probs[i].tolist(),  # convert to list for JSON compatibility
                })
                    
                bar()
                
    pred_df = pd.DataFrame(row)
    
    # Ensure matching dtypes
    df["taxid"] = df["taxid"].astype(int)
    pred_df["taxon"] = pred_df["taxon"].astype(int)

    df["seq"] = df["seq"].astype(str)
    pred_df["seq_id"] = pred_df["seq_id"].astype(str)

    # Merge
    merged_df = df.merge(
        pred_df,
        left_on=["seq", "taxid"],
        right_on=["seq_id", "taxon"],
        how="left",  
        suffixes=("_kraken", "_cnn")
    )
    
    def get_rank_name(taxid):
        try:
            rank_dict = ncbi.get_rank([int(taxid)])
            rank = list(rank_dict.values())[0]
            return rank if rank in CANONICAL_RANKS else None
        except Exception:
            return None
    merged_df["rank"] = merged_df["taxid"].apply(get_rank_name)
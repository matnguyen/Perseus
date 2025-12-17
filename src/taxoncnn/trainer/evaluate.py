import torch
import numpy as np
import logging
from alive_progress import alive_bar

from taxoncnn.utils.constants import CANONICAL_RANKS
from taxoncnn.losses.focal import FocalLoss
from taxoncnn.losses.compute import compute_loss_from_batch
from taxoncnn.trainer.metrics import binary_auroc

logger = logging.getLogger(__name__)

import math
import numpy as np
import torch

def debug_val_loss_spikes(
    model,
    loader,
    device,
    crit,
    *,
    topk=10,
    spike_threshold=None,   # e.g. 0.5 or 1.0; if None, just report topk
    print_per_rank=True,
):
    import math
    """
    Runs validation, computes per-batch masked focal loss, and reports top-k worst batches.
    Also prints diagnostics for any batch exceeding spike_threshold (if provided).

    Assumes:
      - logits: [B,R]
      - batch["y_per_rank"]: [B,R] in {-1,0,1}
      - crit(logits, targets01, mask=mask) averages by mask.sum()
    """

    model.eval()
    bad = []  # list of dicts

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)

            msk = batch.get("mask", None)
            if msk is not None:
                msk = msk.to(device, non_blocking=True)

            extra = torch.log1p(batch["lengths"].to(device).float()).unsqueeze(1).to(torch.float32)

            logits = model(x, mask=msk, extra=extra)          # [B,R]
            y = batch["y_per_rank"].to(device)                # [B,R] {-1,0,1}

            # supervised-label mask
            sup_mask = (y >= 0).float()
            w = int(sup_mask.sum().item())

            # skip batches with no supervised labels
            if w == 0:
                bad.append({
                    "batch": bi,
                    "loss": float("nan"),
                    "w": 0,
                    "reason": "EMPTY_SUPERVISION (all -1)",
                })
                continue

            y01 = y.clamp(min=0).float()
            loss = crit(logits, y01, mask=sup_mask)

            loss_item = float(loss.item())
            finite = math.isfinite(loss_item)

            # collect diagnostics
            logit_min = float(logits.min().item()) if logits.numel() else float("nan")
            logit_max = float(logits.max().item()) if logits.numel() else float("nan")
            logit_mean = float(logits.mean().item()) if logits.numel() else float("nan")
            logit_std = float(logits.std(unbiased=False).item()) if logits.numel() else float("nan")

            # label stats (only on supervised entries)
            y_sup = y[y >= 0]
            pos = int((y_sup == 1).sum().item())
            neg = int((y_sup == 0).sum().item())

            entry = {
                "batch": bi,
                "loss": loss_item,
                "finite": finite,
                "w": w,
                "pos": pos,
                "neg": neg,
                "logit_min": logit_min,
                "logit_max": logit_max,
                "logit_mean": logit_mean,
                "logit_std": logit_std,
            }

            if print_per_rank:
                # per-rank supervision counts
                sup_counts = (y >= 0).sum(dim=0).detach().cpu().tolist()  # length R
                entry["sup_per_rank"] = sup_counts

            # if spike_threshold set, print immediately when exceeded or non-finite
            if (spike_threshold is not None and loss_item > spike_threshold) or (not finite):
                print("\n[VAL SPIKE]")
                for k, v in entry.items():
                    print(f"  {k}: {v}")

                # extra checks
                print("  logits finite:", bool(torch.isfinite(logits).all().item()))
                print("  y unique:", torch.unique(y).detach().cpu().tolist())
                if msk is not None:
                    # how many valid timesteps (if your mask is [B,T] or [B,1,T])
                    if msk.ndim == 3:
                        valid_t = int(msk[:,0,:].sum().item())
                    elif msk.ndim == 2:
                        valid_t = int(msk.sum().item())
                    else:
                        valid_t = int(msk.sum().item())
                    print("  input mask sum (valid bins total):", valid_t)

                # break early if you want to stop on first spike
                # break

            bad.append(entry)

    # Sort by: non-finite first, then highest loss
    def sort_key(d):
        fin = d.get("finite", True)
        l = d.get("loss", float("-inf"))
        # Put non-finite at top
        return (0 if (not fin) else 1, -(l if math.isfinite(l) else 1e9))

    bad_sorted = sorted(bad, key=sort_key)

    print("\n=== TOP WORST VAL BATCHES ===")
    for e in bad_sorted[:topk]:
        print(e)

    return bad_sorted


@torch.no_grad()
def evaluate(model, loader, device, target_mode="any", rank_idx_for_gate=None):
    model.eval()
    crit = FocalLoss(alpha=1, gamma=2)

    total_loss = 0.0
    total_w = 0  # weight = number of contributing labels/examples

    all_logits = []
    all_targets = []

    with torch.no_grad():
        with alive_bar(len(loader), title="Evaluating", force_tty=True) as bar:
            for batch in loader:
                x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)

                msk = batch.get("mask", None)
                if msk is not None:
                    msk = msk.to(device, non_blocking=True)

                extra = torch.log1p(batch["lengths"].to(device).float()).unsqueeze(1).to(torch.float32)

                logits = model(x, mask=msk, extra=extra)

                # ---------- decide weighting + (optional) gating ----------
                if target_mode == "per-rank":
                    ypr = batch["y_per_rank"].to(device)           # [B,R] in {-1,0,1}
                    w = int((ypr >= 0).sum().item())               # #valid labels
                    if w == 0:
                        bar()
                        continue
                    loss = compute_loss_from_batch(logits, batch, device, crit, target_mode, rank_idx_for_gate)

                elif target_mode == "rank":
                    y = batch["y_rank"].to(device)
                    if rank_idx_for_gate is not None:
                        gate = (batch["rank_index"] == int(rank_idx_for_gate)).to(device)
                        if gate.sum().item() == 0:
                            bar()
                            continue
                        # gate BEFORE loss so it matches what you're evaluating
                        y = y[gate]
                        logits_g = logits[gate]
                        w = int(y.numel())
                        loss = crit(logits_g.view_as(y).float(), y.float())  # avoid compute_loss mismatch
                        all_targets.append(y.detach().cpu())
                        all_logits.append(logits_g.detach().cpu())
                    else:
                        w = int(y.numel())
                        loss = compute_loss_from_batch(logits, batch, device, crit, target_mode, rank_idx_for_gate)
                        all_targets.append(y.detach().cpu())
                        all_logits.append(logits.detach().cpu())

                else:  # "any"
                    y = batch["y_any"].to(device)
                    w = int(y.numel())
                    loss = compute_loss_from_batch(logits, batch, device, crit, target_mode, rank_idx_for_gate)
                    all_targets.append(y.detach().cpu())
                    all_logits.append(logits.detach().cpu())

                # ---------- accumulate weighted ----------
                total_loss += loss.item() * w
                total_w += w
                bar()
                
                if total_loss / max(total_w, 1) > 0.5:
                    crit = FocalLoss(alpha=1, gamma=2)
                    _ = debug_val_loss_spikes(
                        model,
                        loader,
                        device,
                        crit,
                        topk=10,
                        spike_threshold=1.0,   # tune this (0.5, 1.0, 2.0)
                    )

    metrics = {"loss": total_loss / max(total_w, 1)}

    if target_mode in ("any", "rank") and all_targets:
        y = torch.cat(all_targets).float().view(-1).numpy()
        s = torch.sigmoid(torch.cat(all_logits)).view(-1).numpy()
        pred = (s >= 0.5).astype(np.int32)
        acc = (pred == y.astype(np.int32)).mean() if y.size else 0.0
        auroc = binary_auroc(y, s)
        metrics.update({"acc": float(acc), "auroc": float(auroc)})

    return metrics


def _collect_scores_per_rank(model, loader, device, calibrators=None, use_calibration=False):
    """Return dict rank_ix -> (y_true, y_score), masking out unknowns (-1)."""
    R = len(CANONICAL_RANKS)
    ys = [ [] for _ in range(R) ]
    ss = [ [] for _ in range(R) ]
    
    model.eval()
    with torch.no_grad():
        with alive_bar(len(loader), title="Collecting scores per rank", force_tty=True) as bar:            
            for batch in loader:
                    x   = batch["x"].to(device, non_blocking=True).float()
                    msk = batch["mask"].to(device, non_blocking=True)
                    extra = torch.log1p(batch["lengths"].to(device).float()).unsqueeze(1)
                    
                    logits = model(x, mask=msk, extra=extra)              # [B, 7]
                    probs  = torch.sigmoid(logits)                        # [B, 7]
        
                    ypr = batch["y_per_rank"].to(device)                  # [B, 7], {-1,0,1}
                    
                    probs_cpu = probs.detach().cpu()
                    ypr_cpu  = ypr.detach().cpu()
                    
                    probs_np = probs_cpu.numpy()
                    
                    for r in range(R):
                        valid = (ypr_cpu[:, r] >= 0)
                        if valid.sum().item() == 0:
                            continue
                        
                        y_r = ypr_cpu[valid, r].to(torch.int32)            # {0,1}
                        ys[r].append(y_r)
                        # ys[r].append(ypr[valid, r].detach().cpu().to(torch.int32))
                        
                        # Apply calibration if provided
                        s_raw = probs_np[valid.numpy(), r]
                        
                        if calibrators is not None and use_calibration:
                            if r in calibrators:
                                iso = calibrators[r]
                            else:
                                rank_name = CANONICAL_RANKS[r]
                                iso = calibrators.get(rank_name, None)
                                
                            if iso is not None:
                                s_calib = iso.predict(s_raw)    # [n_valid]
                            else:
                                s_calib = s_raw
                            ss[r].append(torch.from_numpy(s_calib))
                        else:
                            ss[r].append(torch.from_numpy(s_raw))
                    bar()
    out = {}
    for r in range(len(CANONICAL_RANKS)):
        if ys[r]:
            y = torch.cat(ys[r]).numpy().astype(np.int32)
            s = torch.cat(ss[r]).numpy().astype(np.float32)
        else:
            y = np.array([], dtype=np.int32)
            s = np.array([], dtype=np.float32)
        out[r] = (y, s)
    return out
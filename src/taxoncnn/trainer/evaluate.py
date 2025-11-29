import torch
import numpy as np
import logging
from alive_progress import alive_bar

from taxoncnn.utils.constants import CANONICAL_RANKS
from taxoncnn.losses.focal import FocalLoss
from taxoncnn.losses.compute import compute_loss_from_batch
from taxoncnn.trainer.metrics import _binary_auroc

logger = logging.getLogger(__name__)

@torch.no_grad()
def evaluate(model, loader, device, target_mode="any", rank_idx_for_gate=None):
    """
    Evaluate a model on a validation or test DataLoader

    Computes loss, accuracy, and AUROC for the specified target mode

    Args:
        model (torch.nn.Module): Model to evaluate
        loader (DataLoader): DataLoader providing batches for evaluation
        device (torch.device): Device to run evaluation on
        target_mode (str, optional): Target mode ("any", "rank", or "per-rank"). Defaults to "any"
        rank_idx_for_gate (int or None, optional): If set, only evaluate samples with this rank index. Defaults to None

    Returns:
        dict: Dictionary with evaluation metrics, including "loss", and (if applicable) "acc" and "auroc"
    """
    model.eval()
    crit = FocalLoss(alpha=1, gamma=2)

    total_loss = 0.0
    total_n = 0
    all_logits = []
    all_targets = []

    with alive_bar(len(loader), title="Evaluating", force_tty=True) as bar:
        for batch in loader:
            x   = batch["x"].to(device, non_blocking=True)
            msk = batch["mask"].to(device, non_blocking=True)
            extra = torch.log1p(batch["lengths"].to(device).float()).unsqueeze(1)
            
            # force tensors to fp32 on device
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            # y = y.to(device, non_blocking=True)
            if msk is not None:   msk = msk.to(device, non_blocking=True)           # bool/int ok
            if extra is not None: extra = extra.to(device, dtype=torch.float32, non_blocking=True)

            logits = model(x.float(), mask=msk, extra=extra)
            loss = compute_loss_from_batch(logits, batch, device, crit, target_mode, rank_idx_for_gate)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_n += bs

            if target_mode in ("any","rank"):
                if target_mode == "any":
                    y = batch["y_any"].to(device)
                else:
                    y = batch["y_rank"].to(device)
                    if rank_idx_for_gate is not None:
                        gate = (batch["rank_index"] == int(rank_idx_for_gate)).to(device)
                        y = y[gate]; logits = logits[gate]
                        bs = y.numel()
                        if bs == 0:
                            bar()
                            continue
                all_targets.append(y.detach().cpu())
                all_logits.append(logits.detach().cpu())
            bar()

    metrics = {"loss": total_loss / max(total_n,1)}
    if target_mode in ("any","rank") and all_targets:
        y = torch.cat(all_targets).float().numpy()
        s = torch.sigmoid(torch.cat(all_logits)).numpy()
        pred = (s >= 0.5).astype(np.int32)
        acc = (pred == y.astype(np.int32)).mean() if y.size else 0.0
        auroc = _binary_auroc(y, s)
        metrics.update({"acc": float(acc), "auroc": float(auroc)})
    return metrics


def _collect_scores_per_rank(model, loader, device):
    """Return dict rank_ix -> (y_true, y_score), masking out unknowns (-1)."""
    R = len(CANONICAL_RANKS)
    ys = [ [] for _ in range(R) ]
    ss = [ [] for _ in range(R) ]
    with torch.no_grad():
        with alive_bar(len(loader), title="Collecting scores per rank", force_tty=True) as bar:            
            for batch in loader:
                    x   = batch["x"].to(device, non_blocking=True).float()
                    msk = batch["mask"].to(device, non_blocking=True)
                    extra = torch.log1p(batch["lengths"].to(device).float()).unsqueeze(1)
                    logits = model(x, mask=msk, extra=extra)              # [B, 7]
                    probs  = torch.sigmoid(logits)                        # [B, 7]
        
                    ypr = batch["y_per_rank"].to(device)                  # [B, 7], {-1,0,1}
                    for r in range(R):
                        valid = (ypr[:, r] >= 0)
                        if valid.sum().item() == 0:
                            continue
                        ys[r].append(ypr[valid, r].detach().cpu().to(torch.int32))
                        ss[r].append(probs[valid, r].detach().cpu())
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
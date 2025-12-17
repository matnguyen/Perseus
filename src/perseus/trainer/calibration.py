import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from alive_progress import alive_bar

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from perseus.utils.constants import CANONICAL_RANKS


logger = logging.getLogger(__name__)

@torch.no_grad()
def collect_head_outputs(model, dataloader, device, head_names=None):
    """
    Collect raw sigmoid probabilities and binary labels for each head
    from a held-out calibration dataloader.

    Assumes:
      - model(x) -> dict of logits per head_name
      - batch has: x, labels_dict, where labels_dict[head_name] is 0/1 tensor
    """
    model.eval()
    if head_names is None:
        head_names = CANONICAL_RANKS
    
    scores = {h: [] for h in head_names}
    labels = {h: [] for h in head_names}

    with alive_bar(len(dataloader), title="Collecting head outputs") as bar:
        for batch in dataloader:
            x = batch["x"].to(device)               # [B, C, T]
            msk = batch["mask"].to(device, non_blocking=True)
            extra = torch.log1p(batch["lengths"].to(device).float()).unsqueeze(1)
            
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            if msk is not None:   msk = msk.to(device, non_blocking=True)           # bool/int ok
            if extra is not None: extra = extra.to(device, dtype=torch.float32, non_blocking=True)
            y_per_rank = torch.as_tensor(
                batch["y_per_rank"], dtype=torch.float32
            )                                         # [B, 7]

            logits = model(x, mask=msk, extra=extra)                         # [B, 7]
            probs = torch.sigmoid(logits).cpu().numpy()   # [B, 7]
            y_np = y_per_rank.cpu().numpy()               # [B, 7]

            for j, h in enumerate(head_names):
                scores[h].append(probs[:, j])
                labels[h].append(y_np[:, j])
            bar()

    out = {}
    for h in head_names:
        s = np.concatenate(scores[h])
        l = np.concatenate(labels[h]).reshape(-1)
        out[h] = (s, l)
    return out


def fit_isotonic_per_head(
    head_scores_and_labels: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, IsotonicRegression]:
    """
    Fit an IsotonicRegression calibrator for each head.
    """
    calibrators = {}
    for head, (scores, labels) in head_scores_and_labels.items():
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(scores, labels)
        calibrators[head] = iso
    return calibrators


def save_calibrators(
    calibrators: Dict[str, IsotonicRegression],
    out_dir: Path,
    prefix: str = "calibrator",
) -> None:
    """
    Save each head's calibrator as a pickle.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for head, iso in calibrators.items():
        path = out_dir / f"{prefix}_{head}.pkl"
        with path.open("wb") as f:
            pickle.dump(iso, f)


def load_calibrators(
    in_dir: Path,
    head_names: List[str],
    prefix: str = "calibrator",
) -> Dict[str, IsotonicRegression]:
    """
    Load each head's calibrator from a pickle.
    """
    calibrators = {}
    for head in head_names:
        path = in_dir / f"{prefix}_{head}.pkl"
        with path.open("rb") as f:
            calibrators[head] = pickle.load(f)
    return calibrators

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Masked BCEWithLogits loss with the same call signature as your FocalLoss:
        loss = crit(logits, targets, mask=mask)

    - logits:  [B, ...]
    - targets: same shape, in {0,1} (float)
    - mask:    same shape, 1 for valid, 0 for ignore

    reduction:
      - "mean": sum(masked_loss) / sum(mask)   (or / numel if mask is None)
      - "sum":  sum(masked_loss)
      - "none": returns elementwise loss (masked if mask provided)
    """
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets, mask=None):
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        if mask is not None:
            loss = loss * mask

        if self.reduction == "mean":
            if mask is not None:
                denom = mask.sum().clamp_min(1.0)
            else:
                denom = torch.tensor(loss.numel(), device=loss.device, dtype=loss.dtype)
            return loss.sum() / denom

        elif self.reduction == "sum":
            return loss.sum()

        else:  # "none"
            return loss

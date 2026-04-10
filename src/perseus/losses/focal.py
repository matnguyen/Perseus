import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification tasks

    This loss is designed to address class imbalance by down-weighting easy examples and focusing training on hard negatives

    Args:
        alpha (float, optional): Weighting factor for the rare class. Defaults to 1
        gamma (float, optional): Focusing parameter to reduce the relative loss for well-classified examples. Defaults to 2
        reduction (str, optional): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'. Defaults to 'mean'

    Forward Args:
        logits (Tensor): Predicted logits of shape [B, ...]
        targets (Tensor): Ground truth binary targets of the same shape as logits
        mask (Tensor, optional): Optional mask tensor of the same shape as logits to ignore certain elements

    Returns:
        Tensor: Computed focal loss (scalar if reduction is 'mean' or 'sum', else same shape as input)
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
        
    def forward(self, logits, targets, mask=None):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p   = torch.sigmoid(logits)
        pt  = p * targets + (1 - p) * (1 - targets)
        loss = self.alpha * ((1 - pt) ** self.gamma) * bce
        if mask is not None:
            loss = loss * mask
        if self.reduction == "mean":
            denom = mask.sum().clamp_min(1.0) if mask is not None else torch.tensor(loss.numel(), device=loss.device, dtype=loss.dtype)
            return loss.sum() / denom
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

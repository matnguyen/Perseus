import torch

def compute_loss_from_batch(logits, batch, device, crit, rank_idx_for_gate):
    """
    Compute the loss for a batch based on the target mode

    Args:
        logits (Tensor): Model output logits
        batch (dict): Batch dictionary containing targets and rank indices
        device (torch.device): Device to move tensors to
        crit (callable): Loss function (criterion)
        rank_idx_for_gate (int or None): If set, applies a mask for the specified rank index

    Returns:
        Tensor: Computed loss value
    """
    y = batch["labels_per_rank"].to(device)               # [B,R]
    if logits.ndim == 1:
        raise ValueError("per-rank target requires model out_dim == R")
    mask = (y >= 0).float()                          # [B,R] 1 where known, 0 where unknown
    y01  = y.clamp(min=0).float()                    # map -1 -> 0 (masked out anyway)

    return crit(logits, y01, mask=mask)
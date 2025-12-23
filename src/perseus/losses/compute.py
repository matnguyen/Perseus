import torch

def compute_loss_from_batch(logits, batch, device, crit, target_mode, rank_idx_for_gate):
    """
    Compute the loss for a batch based on the target mode

    Supports three modes:
        - "any": Uses y_any as the target
        - "rank": Uses y_rank as the target, with optional gating by rank index
        - "per-rank": Uses y_per_rank as the target for multi-head outputs

    Args:
        logits (Tensor): Model output logits
        batch (dict): Batch dictionary containing targets and rank indices
        device (torch.device): Device to move tensors to
        crit (callable): Loss function (criterion)
        target_mode (str): Target mode ("any", "rank", or "per-rank")
        rank_idx_for_gate (int or None): If set, applies a mask for the specified rank index

    Returns:
        Tensor: Computed loss value
    """
    if target_mode == "any":
        y = batch["y_any"].to(device).view(-1, 1 if logits.ndim==2 else 1).squeeze(-1)
        return crit(logits.view_as(y), y)

    if target_mode == "rank":
        y = batch["y_rank"].to(device).view(-1, 1 if logits.ndim==2 else 1).squeeze(-1)
        if rank_idx_for_gate is not None:
            gate = (batch["rank_index"] == int(rank_idx_for_gate)).to(device).float()
            return crit(logits.view_as(y), y, mask=gate)
        else:
            return crit(logits.view_as(y), y)

    # per-rank
    y = batch["y_per_rank"].to(device)               # [B,R]
    if logits.ndim == 1:
        raise ValueError("per-rank target requires model out_dim == R")
    mask = (y >= 0).float()                          # [B,R] 1 where known, 0 where unknown
    y01  = y.clamp(min=0).float()                    # map -1 -> 0 (masked out anyway)

    return crit(logits, y01, mask=mask)
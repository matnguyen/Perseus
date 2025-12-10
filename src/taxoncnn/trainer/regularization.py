import torch

def random_bin_masking_batch(x, mask, p=0.1):
    """
    x:    [B, C, T]
    mask: [B, T] (1 = valid, 0 = pad)
    p:    masking probability per valid bin
    """
    B, C, T = x.shape
    valid = mask.bool()                     # [B, T]
    
    # sample random mask
    rand = torch.rand(B, T, device=x.device)
    to_mask = (rand < p) & valid            # [B, T]
    
    # expand to channels
    to_mask_exp = to_mask.unsqueeze(1)      # [B, 1, T]

    # apply masking
    x = x.masked_fill(to_mask_exp, 0.0)

    return x

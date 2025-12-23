import torch

from perseus.utils.constants import CROP_MAX_T

class PadMaskCollateCF:
    """
    Collate function for variable-length channel-first tensors with padding and masking

    Pads each sample in the batch to the maximum sequence length in the batch (or to max_len if specified),
    and creates a mask indicating valid positions for each sample

    Args:
        max_len (int, optional): Maximum sequence length to crop/pad to (default: CROP_MAX_T)
        train (bool, optional): If True, enables random cropping for sequences longer than max_len
    """
    def __init__(self, max_len=CROP_MAX_T, train=True):
        """
        Initialize the collate function

        Args:
            max_len (int, optional): Maximum sequence length to crop/pad to
            train (bool, optional): If True, enables random cropping for sequences longer than max_len
        """
        self.max_len = max_len if train else None
        self.train   = train

    def __call__(self, batch):
        """
        Collate a batch of samples, padding and masking as needed

        Args:
            batch (list[dict]): List of samples, each a dict with keys:
                - "x": Tensor [C, T]
                - "y_any": float
                - "y_rank": float
                - "y_per_rank": Tensor
                - "rank_index": int

        Returns:
            dict: Batch dict with keys:
                - "x": Tensor [B, C, T_max]
                - "mask": BoolTensor [B, 1, T_max]
                - "lengths": IntTensor [B]
                - "y_any": FloatTensor [B]
                - "y_rank": FloatTensor [B]
                - "y_per_rank": Tensor [B, ...]
                - "rank_index": IntTensor [B]
        """
        # --- drop any zero-length samples ---
        keep = []
        dropped = []
        for b in batch:
            x = b["x"]
            if x is None or x.numel() == 0 or x.size(-1) == 0:
                dropped.append((b.get("seq_id", None), b.get("taxon", None)))
            else:
                keep.append(b)

        if len(keep) == 0:
            # Fail loudly with info instead of making T_max=0
            raise RuntimeError(f"All samples in batch have T=0. Dropped={dropped[:5]} (showing up to 5)")

        if dropped:
            # Optional: only print occasionally if too spammy
            print(f"[PadMaskCollateCF] dropped {len(dropped)} zero-length samples, e.g. {dropped[:3]}")

        batch = keep

        xs = [b["x"] for b in batch]
        proc, lens = [], []

        for x in xs:
            T = x.size(-1)

            # deterministic or random crop for long sequences
            if self.max_len is not None and T > self.max_len:
                if self.train:
                    st = torch.randint(0, T - self.max_len + 1, (1,)).item()
                else:
                    st = 0
                x = x[..., st:st + self.max_len]
                T = x.size(-1)
                
            if T == 0:
                raise RuntimeError(
                    f"Unexpected T=0 after cropping. "
                    f"max_len={self.max_len}, train={self.train}, "
                    f"orig_T={x_orig_T if 'x_orig_T' in locals() else '??'}, "
                    f"seq_id={b.get('seq_id', None)}, taxon={b.get('taxon', None)}"
                )

            # extra guard
            if T == 0:
                raise RuntimeError("Unexpected T=0 after cropping. Check max_len and input tensors.")

            proc.append(x)
            lens.append(T)

        T_max = self.max_len if self.max_len is not None else max(lens)
        if T_max is None or T_max <= 0:
            raise RuntimeError(f"Invalid T_max={T_max}. lens={lens[:10]}")

        B, C = len(proc), proc[0].size(0)
        X = torch.zeros(B, C, T_max, dtype=proc[0].dtype)
        M = torch.zeros(B, 1, T_max, dtype=torch.bool)

        for i, x in enumerate(proc):
            Ti = min(x.size(-1), T_max)
            X[i, :, :Ti] = x[..., :Ti]
            M[i, 0, :Ti] = True

        y_any  = torch.tensor([b["y_any"] for b in batch], dtype=torch.float32)
        y_rank = torch.tensor([b["y_rank"] for b in batch], dtype=torch.float32)
        y_pr   = torch.stack([b["y_per_rank"] for b in batch], dim=0)
        rix    = torch.tensor([b["rank_index"] for b in batch], dtype=torch.int64)
        seq_id = [b.get("seq_id", None) for b in batch]
        taxon  = [b.get("taxon", None) for b in batch]
        Ls     = torch.tensor(lens, dtype=torch.int32)

        return {
            "x": X,
            "mask": M,
            "lengths": Ls,
            "y_any": y_any,
            "y_rank": y_rank,
            "y_per_rank": y_pr,
            "rank_index": rix,
            "seq_id": seq_id,
            "taxon": taxon,
        }

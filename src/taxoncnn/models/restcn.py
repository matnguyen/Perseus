import torch
import torch.nn as nn
import torch.nn.functional as F

from taxoncnn.utils.constants import N_CHANNELS
from taxoncnn.models.layers import (
    Bottleneck1D,
    masked_avgpool1d
)

class ResTCN_CF(nn.Module):
    def __init__(self, in_channels=N_CHANNELS, out_dim=1, extra_dim=1, widths=(64,128,256), dilations=(1,2,4,8)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, widths[0], 7, padding=3),
                                  nn.BatchNorm1d(widths[0]), nn.GELU())
        blocks = []; c = widths[0]
        for w in widths:
            blocks += [Bottleneck1D(c, w, dilation=dilations[0], stride=1 if c==w else 2),
                       Bottleneck1D(w, w, dilation=dilations[1], stride=1)]
            c = w
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(c + extra_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, out_dim))

    def forward(self, x, mask=None, extra=None):
        h = self.body(self.stem(x))
        h_vec = masked_avgpool1d(h, mask) if mask is not None else F.adaptive_avg_pool1d(h,1).squeeze(-1)
        if extra is not None:
            h_vec = torch.cat([h_vec, extra], dim=1)
        return self.head(h_vec)  # [B,out_dim]
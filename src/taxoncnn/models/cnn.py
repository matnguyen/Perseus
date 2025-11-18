import torch
import torch.nn as nn
import torch.nn.functional as F

from taxoncnn.utils.constants import N_CHANNELS
from taxoncnn.models.layers import masked_avgpool1d

class CNN1D_CF(nn.Module):
    def __init__(self, in_channels=N_CHANNELS, out_dim=1, extra_dim=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + extra_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x, mask=None, extra=None):   # x:[B,C,T]
        h = self.conv(x)                            # [B,256,T]
        h_vec = masked_avgpool1d(h, mask) if mask is not None else F.adaptive_avg_pool1d(h,1).squeeze(-1)
        if extra is not None:
            h_vec = torch.cat([h_vec, extra], dim=1)
        return self.classifier(h_vec)               # [B,out_dim]
import torch
import torch.nn as nn
import torch.nn.functional as F

from taxoncnn.utils.constants import N_CHANNELS
from taxoncnn.models.layers import masked_avgpool1d

class ConvBlock1D(nn.Module):
    """
    Conv block with optional bottleneck, BatchNorm, ReLU, and Dropout.

    Input:  [B, C_in, T]
    Output: [B, C_out, T]
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        drop: float = 0.1,
        use_bn: bool = True,
        bottleneck: bool = True,
        bottleneck_factor: int = 2,
    ):
        super().__init__()

        layers = []

        if bottleneck and in_channels != out_channels:
            # reduce channels, then expand back out
            mid_channels = max(out_channels // bottleneck_factor, 16)

            # 1x1 bottleneck conv
            layers.append(nn.Conv1d(in_channels, mid_channels, kernel_size=1))
            if use_bn:
                layers.append(nn.BatchNorm1d(mid_channels))
            layers.append(nn.ReLU(inplace=True))

            # main 3x1 conv
            layers.append(
                nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
            if use_bn:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        else:
            # simple non-bottleneck block
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
            if use_bn:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        # dropout at the end of the block
        if drop > 0.0:
            layers.append(nn.Dropout(drop))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNN1D_CF(nn.Module):
    def __init__(
        self,
        in_channels: int = N_CHANNELS,
        out_dim: int = 1,
        extra_dim: int = 1,
        drop_c: float = 0.1,   # conv dropout
        drop_f: float = 0.2,   # fc dropout
        use_bn: bool = True,
        use_bottleneck: bool = True,
    ):
        super().__init__()

        # ---- Convolutional feature extractor ----
        self.conv = nn.Sequential(
            ConvBlock1D(
                in_channels, 64,
                kernel_size=3, padding=1,
                drop=drop_c, use_bn=use_bn, bottleneck=use_bottleneck
            ),
            ConvBlock1D(
                64, 128,
                kernel_size=3, padding=1,
                drop=drop_c, use_bn=use_bn, bottleneck=use_bottleneck
            ),
            ConvBlock1D(
                128, 128,
                kernel_size=3, padding=1,
                drop=drop_c, use_bn=use_bn, bottleneck=use_bottleneck
            ),
            ConvBlock1D(
                128, 256,
                kernel_size=3, padding=1,
                drop=drop_c, use_bn=use_bn, bottleneck=use_bottleneck
            ),
            ConvBlock1D(
                256, 256,
                kernel_size=3, padding=1,
                drop=drop_c, use_bn=use_bn, bottleneck=False  # last block no bottleneck if you like
            ),
        )

        # ---- Classifier head ----
        hidden_in = 256 + extra_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_in, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_f),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_f),

            nn.Linear(64, out_dim),
        )

    def forward(self, x, mask=None, extra=None):
        """
        x:     [B, C, T]
        mask:  [B, T] (optional, 1 for valid, 0 for padded)
        extra: [B, extra_dim] (e.g., log(length) or other features)
        """
        h = self.conv(x)  # [B, 256, T]

        if mask is not None:
            h_vec = masked_avgpool1d(h, mask)   # [B, 256]
        else:
            h_vec = F.adaptive_avg_pool1d(h, 1).squeeze(-1)  # [B, 256]

        if extra is not None:
            h_vec = torch.cat([h_vec, extra], dim=1)  # [B, 256 + extra_dim]

        out = self.classifier(h_vec)  # [B, out_dim]
        return out
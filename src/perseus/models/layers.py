import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_avgpool1d(h: torch.Tensor, m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if m.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        m = m.float()
    num = (h * m).sum(-1)
    den = m.sum(-1).clamp_min(eps)
    return num / den


class Bottleneck1D(nn.Module):
    def __init__(self, c_in, c_out, dilation=1, stride=1):
        super().__init__()
        mid = max(32, c_out // 4)
        self.proj = (nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False)
                     if (c_in != c_out or stride != 1) else nn.Identity())
        self.net = nn.Sequential(
            nn.Conv1d(c_in, mid, 1, stride=stride, bias=False),
            nn.BatchNorm1d(mid), nn.GELU(),
            nn.Conv1d(mid, mid, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(mid), nn.GELU(),
            nn.Conv1d(mid, c_out, 1, bias=False),
            nn.BatchNorm1d(c_out),
        )
    def forward(self, x):
        y = self.net(x); r = self.proj(x)
        if y.size(-1) != r.size(-1):
            T = min(y.size(-1), r.size(-1))
            y, r = y[..., :T], r[..., :T]
        return F.gelu(y + r)

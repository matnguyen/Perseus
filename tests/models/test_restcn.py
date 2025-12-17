import importlib
import torch

MODULE = "perseus.models.restcn"

def get_seq_len_after_conv(model, x):
    with torch.no_grad():
        h = model.stem(x)
        for layer in model.body:
            h = layer(h)
        return h.shape[-1]


def test_restcn_cf_forward_with_extra():
    m = importlib.import_module(MODULE)
    model = m.ResTCN_CF(in_channels=22, out_dim=2, extra_dim=5)
    x = torch.randn(3, 22, 20)
    extra = torch.randn(3, 5)
    out = model(x, extra=extra)
    assert out.shape == (3, 2)


def test_restcn_cf_forward_with_mask_and_extra():
    m = importlib.import_module(MODULE)
    model = m.ResTCN_CF(in_channels=22, out_dim=2, extra_dim=4)
    x = torch.randn(5, 22, 10)
    seq_len = get_seq_len_after_conv(model, x)
    mask = torch.ones(5, 1, seq_len)  # <-- add singleton channel dimension
    extra = torch.randn(5, 4)
    out = model(x, mask=mask, extra=extra)
    assert out.shape == (5, 2)
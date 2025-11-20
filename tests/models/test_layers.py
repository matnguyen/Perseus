import importlib
import torch

MODULE = "taxoncnn.models.layers"

def test_masked_avgpool1d_basic():
    m = importlib.import_module(MODULE)
    h = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    out = m.masked_avgpool1d(h, mask)
    assert torch.allclose(out, torch.tensor([2.0]))


def test_masked_avgpool1d_partial_mask():
    m = importlib.import_module(MODULE)
    h = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    out = m.masked_avgpool1d(h, mask)
    assert torch.allclose(out, torch.tensor([2.0]))


def test_masked_avgpool1d_all_masked():
    m = importlib.import_module(MODULE)
    h = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[0.0, 0.0, 0.0]])
    out = m.masked_avgpool1d(h, mask)
    assert torch.isfinite(out).all()


def test_bottleneck1d_forward_shapes():
    m = importlib.import_module(MODULE)
    layer = m.Bottleneck1D(16, 32)
    x = torch.randn(4, 16, 20)
    y = layer(x)
    assert y.shape == (4, 32, 20)


def test_bottleneck1d_forward_stride():
    m = importlib.import_module(MODULE)
    layer = m.Bottleneck1D(8, 8, stride=2)
    x = torch.randn(2, 8, 16)
    y = layer(x)
    assert y.shape[1] == 8
    assert y.shape[2] == x.shape[2] // 2


def test_bottleneck1d_forward_dilation():
    m = importlib.import_module(MODULE)
    layer = m.Bottleneck1D(8, 8, dilation=2)
    x = torch.randn(2, 8, 16)
    y = layer(x)
    assert y.shape == x.shape
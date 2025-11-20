import importlib
import torch

MODULE = "taxoncnn.losses.compute"

def dummy_loss_fn(logits, targets, mask=None):
    # Simple mean absolute error for testing, supports optional mask
    loss = (logits - targets).abs()
    if mask is not None:
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom
    return loss.mean()


def test_compute_loss_any_mode():
    m = importlib.import_module(MODULE)
    logits = torch.tensor([0.5, 0.2, 0.8])
    batch = {"y_any": torch.tensor([1.0, 0.0, 1.0])}
    device = torch.device("cpu")
    loss = m.compute_loss_from_batch(logits, batch, device, dummy_loss_fn, "any", None)
    assert torch.is_tensor(loss)
    assert loss > 0


def test_compute_loss_rank_mode_no_gate():
    m = importlib.import_module(MODULE)
    logits = torch.tensor([0.5, 0.2, 0.8])
    batch = {"y_rank": torch.tensor([1.0, 0.0, 1.0]), "rank_index": torch.tensor([1, 2, 1])}
    device = torch.device("cpu")
    loss = m.compute_loss_from_batch(logits, batch, device, dummy_loss_fn, "rank", None)
    assert torch.is_tensor(loss)
    assert loss > 0


def test_compute_loss_rank_mode_with_gate():
    m = importlib.import_module(MODULE)
    logits = torch.tensor([0.5, 0.2, 0.8])
    batch = {"y_rank": torch.tensor([1.0, 0.0, 1.0]), "rank_index": torch.tensor([1, 2, 1])}
    device = torch.device("cpu")
    # Only positions with rank_index==1 will be included
    loss = m.compute_loss_from_batch(logits, batch, device, dummy_loss_fn, "rank", 1)
    assert torch.is_tensor(loss)
    assert loss > 0


def test_compute_loss_per_rank_mode():
    m = importlib.import_module(MODULE)
    logits = torch.tensor([[0.5, 0.2], [0.8, 0.1]])
    batch = {"y_per_rank": torch.tensor([[1.0, 0.0], [1.0, 1.0]])}
    device = torch.device("cpu")
    loss = m.compute_loss_from_batch(logits, batch, device, dummy_loss_fn, "per-rank", None)
    assert torch.is_tensor(loss)
    assert loss > 0


def test_compute_loss_per_rank_mode_raises_on_1d_logits():
    m = importlib.import_module(MODULE)
    logits = torch.tensor([0.5, 0.2])
    batch = {"y_per_rank": torch.tensor([[1.0, 0.0], [1.0, 1.0]])}
    device = torch.device("cpu")
    try:
        m.compute_loss_from_batch(logits, batch, device, dummy_loss_fn, "per-rank", None)
    except ValueError as e:
        assert "per-rank target requires model out_dim == R" in str(e)
    else:
        assert False, "Expected ValueError for 1D logits in per-rank mode"
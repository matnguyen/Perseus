import importlib
import torch

MODULE = "perseus.losses.focal"

def test_focal_loss_basic_mean():
    m = importlib.import_module(MODULE)
    loss_fn = m.FocalLoss(alpha=1, gamma=2, reduction='mean')
    logits = torch.tensor([0.0, 2.0, -2.0])
    targets = torch.tensor([0.0, 1.0, 0.0])
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0
    assert loss > 0


def test_focal_loss_sum():
    m = importlib.import_module(MODULE)
    loss_fn = m.FocalLoss(alpha=1, gamma=2, reduction='sum')
    logits = torch.tensor([0.0, 2.0, -2.0])
    targets = torch.tensor([0.0, 1.0, 0.0])
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0
    assert loss > 0


def test_focal_loss_none():
    m = importlib.import_module(MODULE)
    loss_fn = m.FocalLoss(alpha=1, gamma=2, reduction='none')
    logits = torch.tensor([0.0, 2.0, -2.0])
    targets = torch.tensor([0.0, 1.0, 0.0])
    loss = loss_fn(logits, targets)
    assert loss.shape == logits.shape


def test_focal_loss_mask():
    m = importlib.import_module(MODULE)
    loss_fn = m.FocalLoss(alpha=1, gamma=2, reduction='mean')
    logits = torch.tensor([0.0, 2.0, -2.0])
    targets = torch.tensor([0.0, 1.0, 0.0])
    mask = torch.tensor([1.0, 0.0, 1.0])
    loss = loss_fn(logits, targets, mask=mask)
    assert loss.ndim == 0


def test_focal_loss_all_masked():
    m = importlib.import_module(MODULE)
    loss_fn = m.FocalLoss(alpha=1, gamma=2, reduction='mean')
    logits = torch.tensor([0.0, 2.0, -2.0])
    targets = torch.tensor([0.0, 1.0, 0.0])
    mask = torch.tensor([0.0, 0.0, 0.0])
    loss = loss_fn(logits, targets, mask=mask)
    assert loss == 0


def test_focal_loss_perfect_prediction():
    m = importlib.import_module(MODULE)
    loss_fn = m.FocalLoss(alpha=1, gamma=2, reduction='mean')
    logits = torch.tensor([20.0, -20.0])
    targets = torch.tensor([1.0, 0.0])
    loss = loss_fn(logits, targets)
    assert loss < 1e-4
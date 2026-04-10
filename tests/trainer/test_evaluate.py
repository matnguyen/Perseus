import importlib
import numpy as np
import torch
import pytest

MODULE = "perseus.trainer.evaluate"


class DummyBar:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return lambda *a, **k: None

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyEvalModel(torch.nn.Module):
    def forward(self, x, mask=None, extra=None):
        # evaluate() does not inspect logits directly; loss is monkeypatched
        # Return [B, 7] to match expected multi-rank output shape
        return torch.ones((x.shape[0], 7), dtype=torch.float32, device=x.device)


class DummyCollectModel(torch.nn.Module):
    def forward(self, x, mask=None, extra=None):
        # Return fixed logits [B, 7]
        b = x.shape[0]
        out = torch.zeros((b, 7), dtype=torch.float32, device=x.device)
        out[:, 0] = 0.0    # sigmoid = 0.5
        out[:, 1] = 2.0    # sigmoid ~ 0.8808
        out[:, 2] = -2.0   # sigmoid ~ 0.1192
        return out


class DummyCalibrator:
    def predict(self, arr):
        return arr + 0.1


@pytest.mark.dev
def test_evaluate_empty_loader(monkeypatch):
    m = importlib.import_module(MODULE)

    monkeypatch.setattr(m, "alive_bar", DummyBar)

    model = DummyEvalModel()
    loader = []
    device = torch.device("cpu")

    metrics = m.evaluate(model, loader, device)

    assert "loss" in metrics
    assert metrics["loss"] == 0.0


@pytest.mark.dev
def test_evaluate_weighted_loss(monkeypatch):
    m = importlib.import_module(MODULE)

    monkeypatch.setattr(m, "alive_bar", DummyBar)

    # Make loss deterministic and independent of model internals
    def fake_compute_loss_from_batch(logits, batch, device, crit, rank_idx_for_gate):
        return torch.tensor(2.5, dtype=torch.float32)

    monkeypatch.setattr(m, "compute_loss_from_batch", fake_compute_loss_from_batch)

    model = DummyEvalModel()
    device = torch.device("cpu")

    batch = {
        "x": torch.ones((3, 10), dtype=torch.float32),
        "mask": torch.ones((3,), dtype=torch.float32),
        "lengths": torch.tensor([10, 20, 30], dtype=torch.float32),
        # 4 valid entries total
        "labels_per_rank": torch.tensor([
            [1,  0, -1],
            [0, -1, -1],
            [1,  1, -1],
        ], dtype=torch.int64),
    }

    loader = [batch]
    metrics = m.evaluate(model, loader, device)

    assert "loss" in metrics
    # total_loss = 2.5 * 4, total_n = 4 => 2.5
    assert metrics["loss"] == pytest.approx(2.5)


@pytest.mark.dev
def test_collect_scores_per_rank_masks_unknowns(monkeypatch):
    m = importlib.import_module(MODULE)

    monkeypatch.setattr(m, "alive_bar", DummyBar)
    monkeypatch.setattr(m, "CANONICAL_RANKS", [
        "superkingdom", "phylum", "class", "order", "family", "genus", "species"
    ])

    model = DummyCollectModel()
    device = torch.device("cpu")

    batch = {
        "x": torch.ones((3, 5), dtype=torch.float32),
        "mask": torch.ones((3,), dtype=torch.float32),
        "lengths": torch.tensor([10, 20, 30], dtype=torch.float32),
        "labels_per_rank": torch.tensor([
            [ 1,  0, -1, -1, -1, -1, -1],
            [ 0, -1, -1, -1, -1, -1, -1],
            [ 1,  1, -1, -1, -1, -1, -1],
        ], dtype=torch.int64),
    }

    loader = [batch]
    out = m._collect_scores_per_rank(model, loader, device)

    # rank 0: all 3 are valid
    y0, s0 = out[0]
    assert y0.tolist() == [1, 0, 1]
    assert len(s0) == 3
    assert np.allclose(s0, np.array([0.5, 0.5, 0.5], dtype=np.float32), atol=1e-6)

    # rank 1: only rows 0 and 2 are valid
    y1, s1 = out[1]
    assert y1.tolist() == [0, 1]
    assert len(s1) == 2
    assert np.allclose(
        s1,
        np.array([torch.sigmoid(torch.tensor(2.0)).item()] * 2, dtype=np.float32),
        atol=1e-6,
    )

    # rank 2 had only unknowns
    y2, s2 = out[2]
    assert y2.size == 0
    assert s2.size == 0


@pytest.mark.dev
def test_collect_scores_per_rank_with_calibration_by_index(monkeypatch):
    m = importlib.import_module(MODULE)

    monkeypatch.setattr(m, "alive_bar", DummyBar)
    monkeypatch.setattr(m, "CANONICAL_RANKS", [
        "superkingdom", "phylum", "class", "order", "family", "genus", "species"
    ])

    model = DummyCollectModel()
    device = torch.device("cpu")

    batch = {
        "x": torch.ones((2, 5), dtype=torch.float32),
        "mask": torch.ones((2,), dtype=torch.float32),
        "lengths": torch.tensor([10, 20], dtype=torch.float32),
        "labels_per_rank": torch.tensor([
            [1, 0, -1, -1, -1, -1, -1],
            [0, 1, -1, -1, -1, -1, -1],
        ], dtype=torch.int64),
    }

    loader = [batch]
    calibrators = {0: DummyCalibrator()}

    out = m._collect_scores_per_rank(
        model,
        loader,
        device,
        calibrators=calibrators,
        use_calibration=True,
    )

    y0, s0 = out[0]
    assert y0.tolist() == [1, 0]
    assert np.allclose(s0, np.array([0.6, 0.6], dtype=np.float32), atol=1e-6)


@pytest.mark.dev
def test_collect_scores_per_rank_with_calibration_by_rank_name(monkeypatch):
    m = importlib.import_module(MODULE)

    monkeypatch.setattr(m, "alive_bar", DummyBar)
    monkeypatch.setattr(m, "CANONICAL_RANKS", [
        "superkingdom", "phylum", "class", "order", "family", "genus", "species"
    ])

    model = DummyCollectModel()
    device = torch.device("cpu")

    batch = {
        "x": torch.ones((2, 5), dtype=torch.float32),
        "mask": torch.ones((2,), dtype=torch.float32),
        "lengths": torch.tensor([10, 20], dtype=torch.float32),
        "labels_per_rank": torch.tensor([
            [1, 0, -1, -1, -1, -1, -1],
            [0, 1, -1, -1, -1, -1, -1],
        ], dtype=torch.int64),
    }

    loader = [batch]
    calibrators = {"phylum": DummyCalibrator()}

    out = m._collect_scores_per_rank(
        model,
        loader,
        device,
        calibrators=calibrators,
        use_calibration=True,
    )

    y1, s1 = out[1]
    raw = torch.sigmoid(torch.tensor(2.0)).item()
    assert y1.tolist() == [0, 1]
    assert np.allclose(
        s1,
        np.array([raw + 0.1, raw + 0.1], dtype=np.float32),
        atol=1e-6,
    )
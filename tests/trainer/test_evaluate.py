import importlib
import torch

MODULE = "perseus.trainer.evaluate"

class DummyModel(torch.nn.Module):
    def forward(self, x, mask=None, extra=None):
        # Return logits: just sum features for each sample
        return x.sum(dim=1)


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = [
            {
                "x": torch.ones(5, 10),  # batch of 5, 10 features
                "mask": torch.ones(5),
                "lengths": torch.ones(5),
                "y_any": torch.tensor([1, 0, 1, 0, 1], dtype=torch.float32),
                "y_rank": torch.tensor([1, 0, 1, 0, 1], dtype=torch.float32),
                "rank_index": torch.tensor([1, 1, 1, 1, 1]),
            }
        ]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def test_evaluate_any_mode():
    m = importlib.import_module(MODULE)
    model = DummyModel()
    dataset = DummyDataset()
    loader = [dataset[0]]  # Simulate a DataLoader with one batch
    device = torch.device("cpu")
    metrics = m.evaluate(model, loader, device, target_mode="any")
    assert "loss" in metrics
    assert "acc" in metrics
    assert "auroc" in metrics
    assert 0.0 <= metrics["acc"] <= 1.0
    assert 0.0 <= metrics["auroc"] <= 1.0


def test_evaluate_rank_mode():
    m = importlib.import_module(MODULE)
    model = DummyModel()
    dataset = DummyDataset()
    loader = [dataset[0]]
    device = torch.device("cpu")
    metrics = m.evaluate(model, loader, device, target_mode="rank")
    assert "loss" in metrics
    assert "acc" in metrics
    assert "auroc" in metrics


def test_evaluate_rank_mode_with_gate():
    m = importlib.import_module(MODULE)
    model = DummyModel()
    dataset = DummyDataset()
    loader = [dataset[0]]
    device = torch.device("cpu")
    # All rank_index == 1, so gating should not filter out all samples
    metrics = m.evaluate(model, loader, device, target_mode="rank", rank_idx_for_gate=1)
    assert "loss" in metrics
    assert "acc" in metrics
    assert "auroc" in metrics


def test_evaluate_empty_loader():
    m = importlib.import_module(MODULE)
    model = DummyModel()
    loader = []
    device = torch.device("cpu")
    metrics = m.evaluate(model, loader, device, target_mode="any")
    assert "loss" in metrics
    assert metrics["loss"] == 0.0
import importlib
import torch
import os
import pytest

MODULE = "perseus.trainer.train"

class DummyModel(torch.nn.Module):
    def __init__(self, n_ranks=3):
        super().__init__()
        self.linear = torch.nn.Linear(10, n_ranks)
    def forward(self, x, mask=None, extra=None):
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        return self.linear(x)


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n=8, C=10, T=1, n_ranks=3):
        self.n = n
        self.C = C
        self.T = T
        self.n_ranks = n_ranks
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return {
            "x": torch.randn(self.C, self.T),  # shape: [C, T]
            "mask": torch.ones(self.T),        # shape: [T]
            "lengths": torch.ones(1),
            "labels_per_rank": torch.ones(self.n_ranks),  # shape: [n_ranks]
            "rank_index": torch.arange(self.n_ranks),     # shape: [n_ranks]
        }

@pytest.mark.dev
def test_train_runs_and_saves(tmp_path):
    m = importlib.import_module(MODULE)
    model = DummyModel()
    train_loader = torch.utils.data.DataLoader(DummyDataset(4), batch_size=2)
    val_loader = torch.utils.data.DataLoader(DummyDataset(2), batch_size=2)
    device = torch.device("cpu")
    save_path = tmp_path / "model.pt"
    m.train(model, train_loader, val_loader, device, epochs=2, save_path=str(save_path))
    assert save_path.exists()

@pytest.mark.dev
def test_train_best_model_logic(tmp_path):
    m = importlib.import_module(MODULE)
    model = DummyModel()
    train_loader = torch.utils.data.DataLoader(DummyDataset(4), batch_size=2)
    val_loader = torch.utils.data.DataLoader(DummyDataset(2), batch_size=2)
    device = torch.device("cpu")
    save_path = tmp_path / "model.pt"
    # Run for 1 epoch to check that at least one model is saved
    m.train(model, train_loader, val_loader, device, epochs=1, save_path=str(save_path))
    assert save_path.exists()
import importlib
import torch
import os

MODULE = "perseus.trainer.train"

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
    def forward(self, x, mask=None, extra=None):
        return self.linear(x).squeeze(-1)


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n=8):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return {
            "x": torch.randn(10),
            "mask": torch.ones(1),
            "lengths": torch.ones(1),
            "y_any": torch.tensor([1.0]),
            "y_rank": torch.tensor([1.0]),
            "rank_index": torch.tensor([1]),
        }


def test_train_runs_and_saves(tmp_path):
    m = importlib.import_module(MODULE)
    model = DummyModel()
    train_loader = torch.utils.data.DataLoader(DummyDataset(4), batch_size=2)
    val_loader = torch.utils.data.DataLoader(DummyDataset(2), batch_size=2)
    device = torch.device("cpu")
    save_path = tmp_path / "model.pt"
    m.train(model, train_loader, val_loader, device, epochs=2, save_path=str(save_path))
    assert save_path.exists()


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
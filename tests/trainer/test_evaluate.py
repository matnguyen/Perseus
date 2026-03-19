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

@pytest.mark.dev
def test_evaluate_empty_loader():
    m = importlib.import_module(MODULE)
    model = DummyModel()
    loader = []
    device = torch.device("cpu")
    metrics = m.evaluate(model, loader, device)
    assert "loss" in metrics
    assert metrics["loss"] == 0.0
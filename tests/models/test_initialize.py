import importlib
from pathlib import Path

import pytest
import torch

MODULE = "perseus.models.initialize"


class DummyModel:
    def __init__(self):
        self.to_device = None
        self.loaded_state = None
        self.loaded_strict = None
        self.eval_called = False

    def to(self, device):
        self.to_device = device
        return self

    def load_state_dict(self, state, strict=True):
        self.loaded_state = state
        self.loaded_strict = strict

    def eval(self):
        self.eval_called = True
        return self


def test_make_model_constructs_cnn_with_expected_args(monkeypatch):
    m = importlib.import_module(MODULE)

    calls = {}

    class FakeCNN:
        def __init__(self, in_channels, out_dim, extra_dim):
            calls["in_channels"] = in_channels
            calls["out_dim"] = out_dim
            calls["extra_dim"] = extra_dim

        def to(self, device):
            calls["device"] = device
            return self

    monkeypatch.setattr(m, "CNN1D_CF", FakeCNN)
    monkeypatch.setattr(m, "N_CHANNELS", 28)

    device = torch.device("cpu")
    model = m.make_model(out_dim=7, device=device)

    assert calls["in_channels"] == 28
    assert calls["out_dim"] == 7
    assert calls["extra_dim"] == 1
    assert calls["device"] == device
    assert isinstance(model, FakeCNN)


def test_load_model_loads_state_sets_eval_and_moves_to_device(monkeypatch, tmp_path):
    m = importlib.import_module(MODULE)

    model = DummyModel()
    device = torch.device("cpu")
    checkpoint_path = tmp_path / "model.pt"

    fake_state = {"weight": torch.tensor([1.0, 2.0])}

    def fake_torch_load(path, map_location=None):
        assert Path(path) == checkpoint_path
        assert map_location == device
        return fake_state

    monkeypatch.setattr(m.torch, "load", fake_torch_load)

    out = m.load_model(model, checkpoint_path, device)

    assert model.loaded_state == fake_state
    assert model.loaded_strict is True
    assert model.eval_called is True
    assert model.to_device == device
    assert out is model


def test_load_default_model_uses_packaged_checkpoint(monkeypatch, tmp_path):
    m = importlib.import_module(MODULE)

    fake_model = DummyModel()
    fake_checkpoint = tmp_path / "default_model.pt"
    fake_checkpoint.write_text("placeholder")

    calls = {}

    def fake_make_model(out_dim, device):
        calls["make_model_out_dim"] = out_dim
        calls["make_model_device"] = device
        return fake_model

    def fake_load_model(model, checkpoint_path, device):
        calls["load_model_model"] = model
        calls["load_model_checkpoint_path"] = Path(checkpoint_path)
        calls["load_model_device"] = device
        return "loaded_model"

    class FakeTraversable:
        def __truediv__(self, other):
            calls["default_model_file"] = other
            return fake_checkpoint

    class DummyContext:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self.path

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(m, "make_model", fake_make_model)
    monkeypatch.setattr(m, "load_model", fake_load_model)
    monkeypatch.setattr(m.resources, "files", lambda package: FakeTraversable())
    monkeypatch.setattr(m.resources, "as_file", lambda path: DummyContext(path))

    device = torch.device("cpu")
    out = m.load_default_model(out_dim=7, device=device)

    assert out == "loaded_model"
    assert calls["make_model_out_dim"] == 7
    assert calls["make_model_device"] == device
    assert calls["default_model_file"] == m.DEFAULT_MODEL_FILE
    assert calls["load_model_model"] is fake_model
    assert calls["load_model_checkpoint_path"] == fake_checkpoint
    assert calls["load_model_device"] == device


def test_build_optimizer_splits_decay_and_no_decay_params():
    m = importlib.import_module(MODULE)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 3)
            self.bn = torch.nn.BatchNorm1d(3)
            self.extra_bias = torch.nn.Parameter(torch.zeros(3))
            self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)

    model = TinyModel()
    opt = m.build_optimizer(model, lr=1e-3, weight_decay=1e-4)

    assert isinstance(opt, torch.optim.AdamW)
    assert len(opt.param_groups) == 2

    decay_group = opt.param_groups[0]
    no_decay_group = opt.param_groups[1]

    assert decay_group["weight_decay"] == pytest.approx(1e-4)
    assert no_decay_group["weight_decay"] == pytest.approx(0.0)
    assert decay_group["lr"] == pytest.approx(1e-3)
    assert no_decay_group["lr"] == pytest.approx(1e-3)

    decay_ids = {id(p) for p in decay_group["params"]}
    no_decay_ids = {id(p) for p in no_decay_group["params"]}

    assert id(model.linear.weight) in decay_ids
    assert id(model.linear.bias) in no_decay_ids
    assert id(model.bn.weight) in no_decay_ids
    assert id(model.bn.bias) in no_decay_ids
    assert id(model.extra_bias) in no_decay_ids

    assert id(model.frozen) not in decay_ids
    assert id(model.frozen) not in no_decay_ids


@pytest.mark.dev
def test_build_optimizer_handles_model_with_only_no_decay_params():
    m = importlib.import_module(MODULE)

    class OnlyBiasModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(5))

    model = OnlyBiasModel()
    opt = m.build_optimizer(model, lr=5e-4, weight_decay=1e-2)

    assert isinstance(opt, torch.optim.AdamW)
    assert len(opt.param_groups) == 2
    assert len(opt.param_groups[0]["params"]) == 0
    assert len(opt.param_groups[1]["params"]) == 1
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(1e-2)
    assert opt.param_groups[1]["weight_decay"] == pytest.approx(0.0)


def test_build_optimizer_handles_model_with_only_decay_params():
    m = importlib.import_module(MODULE)

    class MatrixOnlyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(3, 4))

        def named_parameters(self, prefix="", recurse=True):
            yield "weight", self.weight

    model = MatrixOnlyModule()
    opt = m.build_optimizer(model, lr=2e-3, weight_decay=3e-4)

    assert isinstance(opt, torch.optim.AdamW)
    assert len(opt.param_groups) == 2
    assert len(opt.param_groups[0]["params"]) == 1
    assert len(opt.param_groups[1]["params"]) == 0
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(3e-4)
    assert opt.param_groups[1]["weight_decay"] == pytest.approx(0.0)
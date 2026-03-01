import importlib
import pytest
import torch

MODULE = "perseus.data.collate"
CONSTANTS_MODULE = "perseus.utils.constants"


def test_padmaskcollate_basic_pad_and_mask():
    """
    Two samples with different T:
      - output X padded to max(T) when max_len=None (train=False)
      - mask True for valid positions, False for padding
      - lengths reflect (possibly cropped) T
    """
    m = importlib.import_module(MODULE)
    constants = importlib.import_module(CONSTANTS_MODULE)

    collate = m.PadMaskCollateCF(max_len=constants.CROP_MAX_T, train=False)  # max_len ignored when train=False

    b0 = {"x": torch.ones(3, 5), "y_per_rank": torch.zeros(7), "rank_index": 1, "seq_id": "s0", "taxon": "t0"}
    b1 = {"x": torch.ones(3, 8), "y_per_rank": torch.ones(7),  "rank_index": 2, "seq_id": "s1", "taxon": "t1"}

    out = collate([b0, b1])

    assert out["x"].shape == (2, 3, 8)
    assert out["mask"].shape == (2, 1, 8)
    assert torch.equal(out["lengths"], torch.tensor([5, 8], dtype=torch.int32))

    # masks
    assert out["mask"][0, 0, :5].all()
    assert (~out["mask"][0, 0, 5:]).all()
    assert out["mask"][1, 0, :8].all()

    # stacked labels
    assert out["y_per_rank"].shape == (2, 7)
    assert torch.equal(out["rank_index"], torch.tensor([1, 2], dtype=torch.int64))
    assert out["seq_id"] == ["s0", "s1"]
    assert out["taxon"] == ["t0", "t1"]


def test_padmaskcollate_drops_zero_length_samples(capsys):
    """
    If batch contains any T=0 samples, they are dropped (with a print).
    """
    m = importlib.import_module(MODULE)

    collate = m.PadMaskCollateCF(max_len=10, train=False)

    good = {"x": torch.ones(2, 4), "y_per_rank": torch.zeros(7), "rank_index": 0, "seq_id": "ok", "taxon": "oktax"}
    bad1 = {"x": torch.empty(2, 0), "y_per_rank": torch.zeros(7), "rank_index": 0, "seq_id": "z0", "taxon": "t0"}
    bad2 = {"x": torch.empty(0),    "y_per_rank": torch.zeros(7), "rank_index": 0, "seq_id": "z1", "taxon": "t1"}

    out = collate([bad1, good, bad2])

    # only good remains
    assert out["x"].shape == (1, 2, 4)
    assert torch.equal(out["lengths"], torch.tensor([4], dtype=torch.int32))
    assert out["seq_id"] == ["ok"]

    printed = capsys.readouterr().out
    assert "dropped" in printed


def test_padmaskcollate_all_zero_length_raises():
    """
    If every sample is T=0, should raise RuntimeError with "All samples" message.
    """
    m = importlib.import_module(MODULE)

    collate = m.PadMaskCollateCF(max_len=10, train=False)
    b0 = {"x": torch.empty(2, 0), "y_per_rank": torch.zeros(7), "rank_index": 0, "seq_id": "a", "taxon": "ta"}
    b1 = {"x": torch.empty(3, 0), "y_per_rank": torch.zeros(7), "rank_index": 0, "seq_id": "b", "taxon": "tb"}

    with pytest.raises(RuntimeError, match=r"All samples in batch have T=0"):
        _ = collate([b0, b1])


def test_padmaskcollate_train_crops_to_max_len_deterministic(monkeypatch):
    """
    When train=True and T > max_len, it crops.
    We monkeypatch torch.randint to make crop start deterministic.
    """
    m = importlib.import_module(MODULE)

    # force crop start st=2
    monkeypatch.setattr(torch, "randint", lambda low, high, size: torch.tensor([2]))

    collate = m.PadMaskCollateCF(max_len=5, train=True)

    x = torch.arange(2 * 10, dtype=torch.float32).reshape(2, 10)  # C=2, T=10
    b = {"x": x, "y_per_rank": torch.zeros(7), "rank_index": 0, "seq_id": "s", "taxon": "t"}

    out = collate([b])

    assert out["x"].shape == (1, 2, 5)
    assert torch.equal(out["lengths"], torch.tensor([5], dtype=torch.int32))
    assert out["mask"][0, 0, :5].all()

    # cropped slice should be x[..., 2:7]
    assert torch.equal(out["x"][0], x[..., 2:7])


def test_padmaskcollate_train_pads_to_max_len_even_if_shorter():
    """
    When train=True, T_max is forced to self.max_len (not max(lens)).
    So if all sequences are shorter than max_len, output is padded to max_len.
    """
    m = importlib.import_module(MODULE)

    collate = m.PadMaskCollateCF(max_len=6, train=True)

    b0 = {"x": torch.ones(1, 4), "y_per_rank": torch.zeros(7), "rank_index": 0}
    b1 = {"x": torch.ones(1, 5), "y_per_rank": torch.zeros(7), "rank_index": 0}

    out = collate([b0, b1])

    assert out["x"].shape == (2, 1, 6)
    assert out["mask"].shape == (2, 1, 6)
    assert torch.equal(out["lengths"], torch.tensor([4, 5], dtype=torch.int32))

    # padding positions False
    assert (~out["mask"][0, 0, 4:]).all()
    assert (~out["mask"][1, 0, 5:]).all()
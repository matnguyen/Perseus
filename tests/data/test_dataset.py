import importlib
import pytest
import json
import os
import tempfile
import shutil
from types import SimpleNamespace

import numpy as np
import torch

MODULE = importlib.import_module("perseus.data.dataset")

def _write_shard(path, *, N=3, C=2, T=5, with_lengths=True):
    """
    Create a tiny channel-first shard dict and torch.save() it.
    """
    x = torch.arange(N * C * T, dtype=torch.float32).reshape(N, C, T)
    y_per_rank = torch.randint(0, 2, (N, 7), dtype=torch.int8)
    rank_index = torch.tensor([0, 1, 2][:N], dtype=torch.int8)

    m = {
        "x": x,
        "y_per_rank": y_per_rank,
        "rank_index": rank_index,
        "seq_id": np.array([f"seq{i}" for i in range(N)], dtype=object),
        "taxon": np.array([f"tax{i}" for i in range(N)], dtype=object),
    }
    if with_lengths:
        m["lengths"] = torch.tensor([T, T - 1, T - 2][:N], dtype=torch.int32)

    torch.save(m, path)


def _args(**kw):
    # minimal arg namespace used by make_loader/build_loader
    base = dict(
        seed=0,
        shards_per_epoch=None,
        samples_per_shard=None,
        val_shards_per_epoch=None,
        val_samples_per_shard=None,
        downcast="none",
        cache_shards=1,
        cpu_float32=False,
        split_dir=None,
        num_workers=0,
        crop_max=None,
        rank_cache=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def temp_dir():
    """Create a temporary directory"""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp)


def test_dataset_init_from_dir_and_len(temp_dir, monkeypatch):
    """
    Dataset initialized from a shard directory:
      - sizes inferred from shard contents (fallback path)
      - __len__ equals total samples across shards
      - offsets behave as expected
    """
    m = importlib.import_module("perseus.data.dataset")

    # make 2 shards of sizes 3 and 2
    _write_shard(os.path.join(temp_dir, "000000.pt"), N=3)
    _write_shard(os.path.join(temp_dir, "000001.pt"), N=2)

    # monkeypatch trainer utils to keep output deterministic + independent of real mapping logic
    monkeypatch.setattr(m, "normalize_y_per_rank_to7", lambda y, ranks: y)
    monkeypatch.setattr(m, "remap_rank_index_to7", lambda ix, ranks: int(ix))

    ds = m.ShardedCFTorchDataset(temp_dir, split_dir=None, split="train", cache_shards=1)

    assert len(ds.paths) == 2
    assert ds.sizes == [3, 2]
    assert ds.offsets == [0, 3, 5]
    assert len(ds) == 5

    # sanity check locate boundaries
    assert ds._locate(0) == (0, 0)
    assert ds._locate(2) == (0, 2)
    assert ds._locate(3) == (1, 0)
    assert ds._locate(4) == (1, 1)


def test_dataset_init_from_manifest_uses_manifest_sizes(temp_dir, monkeypatch):
    """
    Dataset initialized from manifest.json:
      - paths resolved relative to manifest parent
      - sizes pulled from manifest["sizes"]
    """
    m = importlib.import_module("perseus.data.dataset")

    shards_dir = os.path.join(temp_dir, "shards")
    os.makedirs(shards_dir)
    _write_shard(os.path.join(shards_dir, "a.pt"), N=4)
    _write_shard(os.path.join(shards_dir, "b.pt"), N=1)

    mani = {
        "outputs": ["shards/a.pt", "shards/b.pt"],
        "sizes": [4, 1],
    }
    mani_path = os.path.join(temp_dir, "manifest.json")
    with open(mani_path, "w") as f:
        json.dump(mani, f)

    monkeypatch.setattr(m, "normalize_y_per_rank_to7", lambda y, ranks: y)
    monkeypatch.setattr(m, "remap_rank_index_to7", lambda ix, ranks: int(ix))

    ds = m.ShardedCFTorchDataset(mani_path, split_dir=None, split="train", cache_shards=1)

    assert ds.sizes == [4, 1]
    assert len(ds) == 5
    assert all(os.path.isabs(p) for p in ds.paths)


def test_allowed_local_indices_train_vs_val(temp_dir, monkeypatch):
    """
    With split_dir:
      - sizes read from split_dir/sizes.json
      - allowed_local_indices uses valmask_{si:06d}.pt
      - val returns indices where valmask==True
      - train returns indices where valmask==False
    """
    m = importlib.import_module("perseus.data.dataset")

    # 1 shard with N=5
    _write_shard(os.path.join(temp_dir, "000000.pt"), N=5)

    split_dir = os.path.join(temp_dir, "split")
    os.makedirs(split_dir)

    with open(os.path.join(split_dir, "sizes.json"), "w") as f:
        json.dump({"sizes": [5]}, f)

    # mark positions 1 and 3 as validation samples
    valmask = torch.tensor([0, 1, 0, 1, 0], dtype=torch.bool)
    torch.save(valmask, os.path.join(split_dir, "valmask_000000.pt"))

    monkeypatch.setattr(m, "normalize_y_per_rank_to7", lambda y, ranks: y)
    monkeypatch.setattr(m, "remap_rank_index_to7", lambda ix, ranks: int(ix))

    ds_val = m.ShardedCFTorchDataset(temp_dir, split_dir=split_dir, split="val")
    ds_tr = m.ShardedCFTorchDataset(temp_dir, split_dir=split_dir, split="train")

    idx_val = ds_val.allowed_local_indices(0)
    idx_tr = ds_tr.allowed_local_indices(0)

    assert np.array_equal(idx_val, np.array([1, 3], dtype=np.int32))
    assert np.array_equal(idx_tr, np.array([0, 2, 4], dtype=np.int32))

    # cached on second call
    assert np.array_equal(ds_val.allowed_local_indices(0), idx_val)


def test_getitem_fields_and_length_logic(temp_dir, monkeypatch):
    """
    __getitem__ returns:
      - x tensor (optionally float32)
      - y_per_rank mapped by normalize_y_per_rank_to7
      - rank_index mapped by remap_rank_index_to7
      - length uses m["lengths"][j] if present else x.size(-1)
      - seq_id/taxon optional strings
    """
    m = importlib.import_module("perseus.data.dataset")

    _write_shard(os.path.join(temp_dir, "000000.pt"), N=3, C=2, T=7, with_lengths=True)

    # make mapping functions obvious in assertions
    monkeypatch.setattr(m, "normalize_y_per_rank_to7", lambda y, ranks: y.to(torch.float32) + 10.0)
    monkeypatch.setattr(m, "remap_rank_index_to7", lambda ix, ranks: int(ix) + 100)

    ds = m.ShardedCFTorchDataset(temp_dir, split_dir=None, split="train", cache_shards=1)

    out = ds[1]
    assert set(out.keys()) == {"x", "y_per_rank", "rank_index", "length", "seq_id", "taxon"}

    assert isinstance(out["x"], torch.Tensor)
    assert out["x"].shape == (2, 7)

    assert isinstance(out["y_per_rank"], torch.Tensor)
    assert out["y_per_rank"].dtype == torch.float32
    # underlying y_per_rank is int8 in shard; our monkeypatch adds 10.0
    assert torch.all(out["y_per_rank"] >= 10.0)

    assert out["rank_index"] == 101  # raw 1 + 100
    assert out["length"] == 6        # lengths[1] = T-1 = 6
    assert out["seq_id"] == "seq1"
    assert out["taxon"] == "tax1"


def test_shard_cache_eviction_lru(temp_dir, monkeypatch):
    """
    cache_shards=1:
      - accessing two different shards evicts the older one
    """
    m = importlib.import_module("perseus.data.dataset")

    _write_shard(os.path.join(temp_dir, "000000.pt"), N=1)
    _write_shard(os.path.join(temp_dir, "000001.pt"), N=1)

    monkeypatch.setattr(m, "normalize_y_per_rank_to7", lambda y, ranks: y)
    monkeypatch.setattr(m, "remap_rank_index_to7", lambda ix, ranks: int(ix))

    ds = m.ShardedCFTorchDataset(temp_dir, split_dir=None, split="train", cache_shards=1)

    _ = ds[0]  # shard 0
    assert list(ds._cache.keys()) == [0]

    _ = ds[1]  # global idx 1 is shard 1, local 0
    assert list(ds._cache.keys()) == [1]  # shard 0 evicted


def test_build_loader_rank_filter_empty_raises(temp_dir, monkeypatch):
    """
    build_loader with rank_filter:
      - if build_rank_filtered_index returns empty, should SystemExit
    """
    m = importlib.import_module("perseus.data.dataset")

    _write_shard(os.path.join(temp_dir, "000000.pt"), N=2)

    monkeypatch.setattr(m, "normalize_y_per_rank_to7", lambda y, ranks: y)
    monkeypatch.setattr(m, "remap_rank_index_to7", lambda ix, ranks: int(ix))
    monkeypatch.setattr(m, "build_rank_filtered_index", lambda *a, **k: ([], {}))

    args = _args(split_dir=None, num_workers=0)
    with pytest.raises(SystemExit):
        _ds, _ld = m.build_loader(args, temp_dir, batch_size=2, train_flag=True, val_flag=False, rank_filter="species")
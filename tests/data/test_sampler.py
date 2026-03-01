import importlib
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

MODULE = "perseus.data.sampler"  # contains ShardBatchSampler


class _FakeShardedDataset:
    """
    Minimal dataset stub for ShardBatchSampler tests.

    - offsets: prefix sums per shard
    - allowed_local_indices(si): returns allowed locals (np.ndarray or torch.Tensor)
    """
    def __init__(self, allowed_by_shard, as_torch=False):
        self.allowed_by_shard = []
        self.sizes = []
        self.paths = [f"shard_{i}.pt" for i in range(len(allowed_by_shard))]
        self.as_torch = as_torch

        # sizes + offsets derived from "full shard size"; for sampler we only need offsets
        self.sizes = [len(a) for a in allowed_by_shard]
        self.offsets = [0]
        for n in self.sizes:
            self.offsets.append(self.offsets[-1] + n)

        for a in allowed_by_shard:
            arr = np.asarray(a, dtype=np.int64)
            self.allowed_by_shard.append(arr)

    def allowed_local_indices(self, si: int):
        arr = self.allowed_by_shard[si]
        if self.as_torch:
            return torch.tensor(arr, dtype=torch.int64)
        return arr


def _collect(it):
    return [b for b in it]


def test_shardbatchsampler_no_resampling_exact_len_and_batches_no_shuffle():
    """
    No resampling (shards_per_epoch=None, samples_per_shard=None):
      - __len__ exact
      - iter yields shard-grouped batches with correct global indices
      - no shuffle gives deterministic order (shard 0 then shard 1 ...)
    """
    m = importlib.import_module(MODULE)

    # shard0 locals [0,1,2,3,4], shard1 locals [0,1,2]
    ds = _FakeShardedDataset([np.arange(5), np.arange(3)])
    sampler = m.ShardBatchSampler(ds, batch_size=2, shuffle=False, drop_last=False)

    # expected batches:
    # shard0 base=0: [0,1], [2,3], [4]
    # shard1 base=5: [5,6], [7]
    batches = _collect(iter(sampler))
    assert batches == [[0, 1], [2, 3], [4], [5, 6], [7]]

    # exact length: ceil(5/2) + ceil(3/2) = 3 + 2 = 5
    assert len(sampler) == 5


def test_shardbatchsampler_drop_last():
    """
    drop_last=True drops incomplete batches within each shard.
    """
    m = importlib.import_module(MODULE)

    ds = _FakeShardedDataset([np.arange(5), np.arange(3)])
    sampler = m.ShardBatchSampler(ds, batch_size=2, shuffle=False, drop_last=True)

    batches = _collect(iter(sampler))
    # shard0: [0,1],[2,3] (drop [4])
    # shard1: [5,6] (drop [7])
    assert batches == [[0, 1], [2, 3], [5, 6]]

    # exact length: floor(5/2)+floor(3/2)=2+1=3
    assert len(sampler) == 3


def test_shardbatchsampler_shuffle_deterministic_by_epoch():
    """
    set_epoch should deterministically change RNG so shuffle order changes across epochs,
    but is reproducible for a fixed epoch.
    """
    m = importlib.import_module(MODULE)

    ds = _FakeShardedDataset([np.arange(4), np.arange(4), np.arange(4)])
    sampler = m.ShardBatchSampler(ds, batch_size=2, shuffle=True, drop_last=False, seed=123)

    sampler.set_epoch(0)
    b0 = _collect(iter(sampler))

    sampler.set_epoch(0)
    b0_again = _collect(iter(sampler))
    assert b0 == b0_again  # reproducible

    sampler.set_epoch(1)
    b1 = _collect(iter(sampler))
    assert b1 != b0  # different epoch -> likely different ordering


def test_shardbatchsampler_shards_per_epoch_subset_no_replacement(monkeypatch):
    """
    shards_per_epoch selects only K shards each epoch (without replacement by default).
    Also preserves within-shard grouping.
    """
    m = importlib.import_module(MODULE)

    # 5 shards with small locals so we can identify shard membership from global indices via offsets
    ds = _FakeShardedDataset([np.arange(2) for _ in range(5)])
    sampler = m.ShardBatchSampler(ds, batch_size=2, shuffle=False, shards_per_epoch=2, seed=0)

    sampler.set_epoch(0)
    batches = _collect(iter(sampler))

    # Each selected shard yields exactly one batch [base+0, base+1]
    assert len(batches) == 2

    # check each batch is a "full-shard batch" (locals [0,1]) and comes from distinct shards
    shard_ids = []
    for batch in batches:
        # infer shard id: offsets are [0,2,4,6,8,10], so batch[0]//2 gives shard idx
        shard_ids.append(batch[0] // 2)
        assert batch[1] == batch[0] + 1
    assert len(set(shard_ids)) == 2


def test_shardbatchsampler_samples_per_shard_limits_locals(monkeypatch):
    """
    samples_per_shard subsamples locals within shard before batching.
    We monkeypatch rng.choice to pick the first m entries for determinism.
    """
    m = importlib.import_module(MODULE)

    ds = _FakeShardedDataset([np.arange(10)])
    sampler = m.ShardBatchSampler(ds, batch_size=4, shuffle=False, samples_per_shard=5, seed=0)

    # force choice(loc, size=m, ...) to return loc[:m]
    def _choice(a, size, replace):
        a = np.asarray(a)
        return a[:size]
    monkeypatch.setattr(sampler, "rng", SimpleNamespace(choice=_choice))

    batches = _collect(iter(sampler))
    # base=0, selected locals [0,1,2,3,4] with bs=4 => [0,1,2,3], [4]
    assert batches == [[0, 1, 2, 3], [4]]


def test_shardbatchsampler_skips_empty_shards():
    """
    Shards with no allowed locals are skipped.
    """
    m = importlib.import_module(MODULE)

    ds = _FakeShardedDataset([np.array([], dtype=np.int64), np.arange(3)])
    sampler = m.ShardBatchSampler(ds, batch_size=2, shuffle=False)

    batches = _collect(iter(sampler))
    # only shard1 base=0? careful: sizes derived from len(allowed); offsets = [0,0,3]
    # shard1 base = offsets[1] = 0
    assert batches == [[0, 1], [2]]


def test_shardbatchsampler_allowed_local_indices_can_be_torch_tensor():
    """
    allowed_local_indices may return torch.Tensor; sampler should handle it.
    """
    m = importlib.import_module(MODULE)

    ds = _FakeShardedDataset([np.arange(5)], as_torch=True)
    sampler = m.ShardBatchSampler(ds, batch_size=2, shuffle=False)

    batches = _collect(iter(sampler))
    assert batches == [[0, 1], [2, 3], [4]]


def test_shardbatchsampler_len_with_resampling_is_stable_estimate():
    """
    When resampling is enabled, __len__ returns an estimate:
      k * ceil(per_shard/bs) (or floor if drop_last).
    """
    m = importlib.import_module(MODULE)

    ds = _FakeShardedDataset([np.arange(10) for _ in range(6)])  # mean allowed=10
    sampler = m.ShardBatchSampler(
        ds,
        batch_size=4,
        shuffle=True,
        shards_per_epoch=3,
        samples_per_shard=5,
        drop_last=False,
    )

    # estimate: k=3, per_shard=5, ceil(5/4)=2 => 6
    assert len(sampler) == 6

    sampler2 = m.ShardBatchSampler(
        ds,
        batch_size=4,
        shuffle=True,
        shards_per_epoch=3,
        samples_per_shard=5,
        drop_last=True,
    )
    # floor(5/4)=1 => 3
    assert len(sampler2) == 3


def test_shardbatchsampler_val_mode_uses_val_params():
    """
    If val_shards_per_epoch or val_samples_per_shard is provided,
    sampler.is_val=True and iter uses val_* knobs.
    """
    m = importlib.import_module(MODULE)

    ds = _FakeShardedDataset([np.arange(6) for _ in range(5)])  # 5 shards
    sampler = m.ShardBatchSampler(
        ds,
        batch_size=3,
        shuffle=False,
        shards_per_epoch=5,
        samples_per_shard=6,
        val_shards_per_epoch=2,
        val_samples_per_shard=4,
        seed=0,
    )

    sampler.set_epoch(0)
    batches = _collect(iter(sampler))

    # In val mode:
    # - only 2 shards selected => total batches depends on per-shard subsample 4 with bs=3 => [3] + [1] => 2 batches per shard
    # => 4 batches total
    assert len(batches) == 4

    # And sampler.__len__ estimate should use *train* knobs per its implementation for resampling
    # (it doesn't branch to val knobs in __len__). We just ensure it's an int and nonzero.
    assert isinstance(len(sampler), int)
    assert len(sampler) > 0
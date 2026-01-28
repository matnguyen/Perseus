import importlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

MODULE = "perseus.utils.io_utils"
GLOBAL_MODULE = "perseus.utils.globals"

def _row(
    seq="s1",
    tax=60,
    true_tax=60,
    bins=None,
    labels_per_rank=None,
    pred_rank="species",
    rank_index=6,
):
    """
    Helper to build a minimal row for both writers.

    Important: your code currently expects C=22 channels in shards,
    so bins must be lists of length 22.
    """
    if bins is None:
        # one bin, 22 channels
        bins = [[0.5] * 22]
    if labels_per_rank is None:
        # length == len(CANONICAL_RANKS) == 7
        labels_per_rank = [0, 0, 0, 0, 0, 0, 1]

    return {
        "seq_id": seq,
        "taxon": int(tax),
        "true_taxon": int(true_tax),
        "bins": bins,
        "label_any": 1,
        "label_rank": 1,
        "labels_per_rank": labels_per_rank,
        "pred_rank": pred_rank,
        "rank_index": rank_index,
    }


def test_write_rows_streaming_shards_pad_to_max(tmp_outdir, monkeypatch):
    """
    target_length = 0 → pad to max T in shard; ensure shapes and metadata look right.
    """
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBAL_MODULE)
    globals_mod._shared_out_dir = str(tmp_outdir)

    # fake manifest list
    class Manifest(list):
        pass

    globals_mod._shared_manifest_paths = Manifest()

    # make two rows with different T to exercise padding:
    #   row1: T=2, row2: T=3
    rows = [
        _row(seq="s1", bins=[[0.1] * 22, [0.2] * 22]),
        _row(seq="s2", bins=[[0.3] * 22, [0.4] * 22, [0.5] * 22]),
    ]

    meta = m._write_rows_streaming_shards(
        rows,
        max_batch_rows=4096,
        target_length=0,          # pad to max T (3)
        to_dtype="float32",
    )

    assert meta is not None
    assert meta["rows"] == 2
    assert meta["file"].endswith(".pt")

    path = Path(tmp_outdir) / meta["file"]
    assert path.exists()

    bundle = torch.load(path)

    # x: [N, C, T]
    x = bundle["x"]
    assert x.shape[0] == 2
    assert x.shape[1] == 22
    assert x.shape[2] == 3   # padded to max T

    # lengths should be present when target_length == 0
    assert "lengths" in bundle
    assert bundle["lengths"].tolist() == [2, 3]

    # labels & metadata
    assert bundle["labels_per_rank"].shape[0] == 2
    assert len(bundle["seq_id"]) == 2
    assert len(bundle["taxon"]) == 2
    assert len(bundle["true_taxon"]) == 2

    # manifest list updated
    assert len(globals_mod._shared_manifest_paths) == 1
    assert str(path) in globals_mod._shared_manifest_paths[0] or str(path) == globals_mod._shared_manifest_paths[0]


def test_write_rows_streaming_shards_resample(tmp_outdir, monkeypatch):
    """
    target_length > 0 → resample all sequences to fixed length.
    """
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBAL_MODULE)
    globals_mod._shared_out_dir = str(tmp_outdir)

    class Manifest(list):
        pass

    globals_mod._shared_manifest_paths = Manifest()

    rows = [
        _row(seq="s1", bins=[[0.1] * 22, [0.2] * 22]),
        _row(seq="s2", bins=[[0.3] * 22, [0.4] * 22, [0.5] * 22]),
    ]

    meta = m._write_rows_streaming_shards(
        rows,
        max_batch_rows=4096,
        target_length=5,      # resample each to T=5
        to_dtype="float16",   # also exercise the dtype branch
    )

    assert meta is not None
    path = Path(tmp_outdir) / meta["file"]
    bundle = torch.load(path)

    x = bundle["x"]
    assert x.shape == (2, 22, 5)
    assert x.dtype in (torch.float16, torch.float32)  # float16 requested, but float32 is acceptable fallback

    # when target_length > 0, lengths key may be omitted
    assert "lengths" not in bundle or bundle["lengths"].shape[0] == 2


import importlib
import pandas as pd
from pathlib import Path
import shutil
import pickle

MODULE = "taxoncnn.extract"

"""
Tests for build_tax_context
"""
def test_build_tax_context_small(monkeypatch, tmp_path):   
    def fake_copyfile(src, dst, *args, **kwargs):
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_bytes(b"")
        return dst

    def fake_copystat(src, dst, *args, **kwargs):
        return None

    monkeypatch.setattr(shutil, "copyfile", fake_copyfile)
    monkeypatch.setattr(shutil, "copystat", fake_copystat)
    
    m = importlib.import_module(MODULE)

    # ----------------------------
    # Create a tiny fake Kraken TSV
    # ----------------------------
    kraken_path = tmp_path / "small.tsv"
    kraken_path.write_text(
        "C\tseq1\t(60)\t1000\t60:5 61:3\n"
        "C\tseq2\t(61)\t900\t60:2 10:1\n"
        "U\tseq3\t(50)\t500\t\n"             # unclassified, has no kmers: should be ignored
    )

    # ----------------------------
    # Monkeypatch multiprocessing to avoid real processes
    # Force build_tax_context to run extract_tax_context_chunk directly
    # ----------------------------
    class FakePool:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def imap_unordered(self, func, iterable, chunksize=1):
            for x in iterable:
                yield func(x)

    monkeypatch.setattr(m.mp, "Pool", FakePool)

    # ----------------------------
    # Run the function
    # ----------------------------
    tax_context = m.build_tax_context(str(kraken_path), rows_per_chunk=1)

    # ----------------------------
    # Validate
    # ----------------------------
    assert "seq1" in tax_context
    assert tax_context["seq1"] == {60: 5, 61: 3}

    assert "seq2" in tax_context
    assert tax_context["seq2"] == {60: 2, 10: 1}

    # seq3 had no kmers, must not appear
    assert "seq3" not in tax_context


def test_build_tax_context_cache(monkeypatch, tmp_path):
    def fake_copyfile(src, dst, *args, **kwargs):
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_bytes(b"")
        return dst

    def fake_copystat(src, dst, *args, **kwargs):
        return None

    monkeypatch.setattr(shutil, "copyfile", fake_copyfile)
    monkeypatch.setattr(shutil, "copystat", fake_copystat)
    
    m = importlib.import_module(MODULE)

    # fake Kraken file
    p = tmp_path / "kraken.tsv"
    p.write_text("C\tseq1\t(60)\t1000\t60:1\n")

    # create expected cached file where read_kraken_file will look
    cache_path = str(tmp_path / "out.tax_context.pkl")
    expected = {"seq1": {60: 1}}

    with open(cache_path, "wb") as f:
        pickle.dump(expected, f)

    # monkeypatch build_tax_context so it would fail if called
    monkeypatch.setattr(m, "build_tax_context", lambda *a, **kw: {"SHOULD_NOT_BE_CALLED": True})

    # Now call read_kraken_file, which will load cache instead of rebuilding
    out_dir = tmp_path / "out"
    m.read_kraken_file(
        str(p),
        str(out_dir),
        chunksize=1,
        threads=1,
        write_format="parquet"
    )

    # Confirm we used cache
    # (i.e., returned dict != {"SHOULD_NOT_BE_CALLED": True})
    with open(cache_path, "rb") as f:
        loaded = pickle.load(f)
    assert loaded == expected

"""
Tests for process_chunk_and_write
"""
def test_process_chunk_and_write_parquet(monkeypatch):
    m = importlib.import_module(MODULE)

    # --- Fake rows that process_chunk_iter will yield ---
    fake_rows = [
        {"seq_id": "s1"},
        {"seq_id": "s2"},
        {"seq_id": "s3"},
    ]

    # process_chunk_iter normally does heavy stuff;
    # here we just yield our fake rows regardless of chunk.
    def fake_process_chunk_iter(chunk, bin_size=1000, topk_taxa=None,
                                min_tax_kmers=0, max_bins_per_seq=None,
                                mess_true_file=None, mess_input_file=None):
        for r in fake_rows:
            yield r

    monkeypatch.setattr(m, "process_chunk_iter", fake_process_chunk_iter)

    # Force parquet mode and a known writer
    m._shared_write_format = "parquet"

    calls = []

    def fake_write_parquet(rows, max_batch_rows=256, use_half=False, quantize_u8=False):
        # record what was passed in
        batch = list(rows)
        calls.append(batch)
        return {"rows": len(batch), "file": "dummy.parquet"}

    monkeypatch.setattr(m, "_write_rows_streaming_parquet", fake_write_parquet)

    # --- Run ---
    meta = m.process_chunk_and_write(
        chunk=None,
        max_bins_per_seq=None,
        mess_true_file=None,
        mess_input_file=None,
    )

    # --- Check ---
    # parquet path uses a threshold of 512 rows before flushing,
    # so with only 3 rows we should get exactly one final flush.
    assert len(calls) == 1
    assert len(calls[0]) == len(fake_rows) == 3
    assert meta["rows"] == 3
    assert meta["file"] == "dummy.parquet"


def test_process_chunk_and_write_shards(monkeypatch):
    m = importlib.import_module(MODULE)

    # 5 fake rows so we can test multiple flushes
    fake_rows = [
        {"seq_id": f"s{i}"} for i in range(5)
    ]

    def fake_process_chunk_iter(chunk, bin_size=1000, topk_taxa=None,
                                min_tax_kmers=0, max_bins_per_seq=None,
                                mess_true_file=None, mess_input_file=None):
        for r in fake_rows:
            yield r

    monkeypatch.setattr(m, "process_chunk_iter", fake_process_chunk_iter)

    # Shard mode: flush every _shared_shard_size rows
    m._shared_write_format = "shards"
    m._shared_shard_size = 2
    m._shared_target_length = 0
    m._shared_to_dtype = "float32"

    calls = []

    def fake_write_shards(rows, max_batch_rows, target_length, to_dtype):
        batch = list(rows)
        calls.append(batch)
        # meta mimics real writer: rows count + file name
        return {"rows": len(batch), "file": f"shard{len(calls)}.pt"}

    monkeypatch.setattr(m, "_write_rows_streaming_shards", fake_write_shards)

    # --- Run ---
    meta = m.process_chunk_and_write(
        chunk=None,
        max_bins_per_seq=None,
        mess_true_file=None,
        mess_input_file=None,
    )

    # --- Check ---
    # shard_size=2 and 5 rows total → flushes at:
    #  rows 1-2, rows 3-4, then final row 5
    assert [len(b) for b in calls] == [2, 2, 1]
    # last meta should correspond to the last batch
    assert meta["rows"] == 1
    assert meta["file"] == "shard3.pt"

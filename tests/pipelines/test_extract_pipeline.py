import importlib
from pathlib import Path
import shutil
import pickle
import pytest

MODULE = "perseus.commands.extract"

@pytest.mark.pipeline
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
        threads=1
    )

    # Confirm we used cache
    # (i.e., returned dict != {"SHOULD_NOT_BE_CALLED": True})
    with open(cache_path, "rb") as f:
        loaded = pickle.load(f)
    assert loaded == expected
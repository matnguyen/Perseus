import importlib
import pandas as pd
import pytest

MODULE = "perseus.features.processing"
GLOBALS_MODULE = "perseus.utils.globals"


class DummyBar:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return lambda *a, **k: None

    def __exit__(self, exc_type, exc, tb):
        return False


"""
Tests for parse_kmers and iter_kmer_tokens
"""
def test_parse_kmers_basic():
    m = importlib.import_module(MODULE)

    s = "60:5 61:3 10:1 A:10 0:240"
    out = m.parse_kmers(s)
    assert out == ["60:5", "61:3", "10:1", "A:10", "0:240"]


def test_parse_kmers_ignore_bad_tokens():
    m = importlib.import_module(MODULE)

    s = "60:5 junk 61:3 nope 10:1"
    out = m.parse_kmers(s)
    assert out == ["60:5", "61:3", "10:1"]


def test_parse_kmers_empty_string():
    m = importlib.import_module(MODULE)

    out = m.parse_kmers("")
    assert out == []


def test_parse_kmers_nan():
    m = importlib.import_module(MODULE)

    out = m.parse_kmers(pd.NA)
    assert out == []


def test_parse_kmers_mixed_whitespace():
    m = importlib.import_module(MODULE)

    s = "   60:5   61:3\t\t10:1   "
    out = m.parse_kmers(s)
    assert out == ["60:5", "61:3", "10:1"]


def test_iter_kmer_tokens_basic():
    m = importlib.import_module(MODULE)

    s = "60:5 61:3 10:1"
    toks = list(m.iter_kmer_tokens(s))
    assert toks == [(60, 5), (61, 3), (10, 1)]


def test_iter_kmer_tokens_ignores_junk():
    m = importlib.import_module(MODULE)

    s = "60:5 junk 61:x 10:1 foo:bar 999:7"
    toks = list(m.iter_kmer_tokens(s))
    assert toks == [(60, 5), (10, 1), (999, 7)]


def test_iter_kmer_tokens_empty_and_none():
    m = importlib.import_module(MODULE)

    assert list(m.iter_kmer_tokens("")) == []
    assert list(m.iter_kmer_tokens(None)) == []
    assert list(m.iter_kmer_tokens(123)) == []


"""
Tests for extract_tax_counts and extract_tax_context_chunk
"""
def test_extract_tax_counts_basic(monkeypatch):
    m = importlib.import_module(MODULE)

    m.normalize_taxid.cache_clear()
    monkeypatch.setattr(m, "normalize_taxid", lambda x: int(x))

    kmer_list = ["60:5", "61:3", "60:2"]
    out = m.extract_tax_counts(kmer_list)

    assert out[60] == 7
    assert out[61] == 3
    assert len(out) == 2


def test_extract_tax_counts_with_bad_tokens(monkeypatch):
    m = importlib.import_module(MODULE)

    monkeypatch.setattr(m, "normalize_taxid", lambda x: int(x))

    kmer_list = ["60:5", "badtoken", "61:x", "abc:2"]
    out = m.extract_tax_counts(kmer_list)

    assert out == {60: 5}


def test_extract_tax_counts_empty(monkeypatch):
    m = importlib.import_module(MODULE)

    monkeypatch.setattr(m, "normalize_taxid", lambda x: int(x))

    out = m.extract_tax_counts([])
    assert out == {}


def test_extract_tax_counts_normalization(monkeypatch):
    m = importlib.import_module(MODULE)

    def fake_norm(x):
        return 50 if int(x) in (60, 61) else int(x)

    monkeypatch.setattr(m, "normalize_taxid", fake_norm)

    kmer_list = ["60:5", "61:3"]
    out = m.extract_tax_counts(kmer_list)

    assert out == {50: 8}


def test_extract_tax_counts_skips_nonpositive_counts(monkeypatch):
    m = importlib.import_module(MODULE)

    monkeypatch.setattr(m, "normalize_taxid", lambda x: int(x))

    kmer_list = ["60:5", "61:0", "63:2"]
    out = m.extract_tax_counts(kmer_list)

    assert out == {60: 5, 63: 2}


def test_extract_tax_context_chunk_basic(monkeypatch):
    m = importlib.import_module(MODULE)

    m.normalize_taxid.cache_clear()
    monkeypatch.setattr(m, "normalize_taxid", lambda x: int(x))

    df = pd.DataFrame({
        "ID": ["seq1", "seq2", "seq3"],
        "Taxonomy": ["(60)", "(61)", "(50)"],
        "Kmers": [
            "60:5 61:3",
            "60:2 60:3 10:1",
            None,
        ],
    })

    out = m.extract_tax_context_chunk(df)

    assert "seq1" in out
    assert out["seq1"] == {60: 5, 61: 3}

    assert "seq2" in out
    assert out["seq2"] == {60: 5, 10: 1}

    assert "seq3" not in out


def test_extract_tax_context_chunk_empty_df():
    m = importlib.import_module(MODULE)

    df = pd.DataFrame(columns=["ID", "Taxonomy", "Kmers"])
    out = m.extract_tax_context_chunk(df)

    assert out == {}


"""
Tests for add_to_bins
"""
def test_add_to_bins_distribution():
    m = importlib.import_module(MODULE)

    bins = {}
    pos = m.add_to_bins(bins, bin_size=10, taxid=60, count=25, cur_pos=0)

    assert pos == 25
    assert bins[0][60] == 10
    assert bins[1][60] == 10
    assert bins[2][60] == 5


def test_add_to_bins_nonzero_start_position():
    m = importlib.import_module(MODULE)

    bins = {}
    pos = m.add_to_bins(bins, bin_size=10, taxid=7, count=8, cur_pos=7)

    assert pos == 15
    assert bins[0][7] == 3
    assert bins[1][7] == 5


"""
Tests for build_tax_context
"""

def test_build_tax_context_single_thread(monkeypatch):
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    chunk1 = pd.DataFrame({
        "ID": ["s1"],
        "Taxonomy": ["(60)"],
        "Kmers": ["60:5 61:1"],
    })
    chunk2 = pd.DataFrame({
        "ID": ["s2"],
        "Taxonomy": ["(61)"],
        "Kmers": ["61:3"],
    })

    monkeypatch.setattr(m.pd, "read_csv", lambda *a, **k: iter([chunk1, chunk2]))
    monkeypatch.setattr(m, "effective_nprocs", lambda: 1)
    monkeypatch.setattr(m, "get_ncbi", lambda db_path: "fake_ncbi")
    monkeypatch.setattr(m, "alive_bar", DummyBar)

    def fake_extract_tax_context_chunk(chunk):
        return {row.ID: {"seen": 1} for row in chunk.itertuples(index=False)}

    monkeypatch.setattr(m, "extract_tax_context_chunk", fake_extract_tax_context_chunk)

    out = m.build_tax_context("fake.tsv", "fake_db", threads=1)

    assert out == {"s1": {"seen": 1}, "s2": {"seen": 1}}
    assert globals_mod.NCBI == "fake_ncbi"


"""
Tests for process_chunk_iter
"""
def test_process_chunk_iter_empty_chunk():
    m = importlib.import_module(MODULE)

    df = pd.DataFrame(columns=["Classified", "ID", "Length", "Kmers", "Taxonomy"])
    out = list(m.process_chunk_iter(df))

    assert out == []


def test_process_chunk_iter_no_classified_rows():
    m = importlib.import_module(MODULE)

    df = pd.DataFrame({
        "Classified": ["U", "U"],
        "ID": ["s1", "s2"],
        "Length": [100, 100],
        "Kmers": ["60:5", "61:3"],
        "Taxonomy": ["60", "61"],
    })

    out = list(m.process_chunk_iter(df))

    assert out == []


def test_process_chunk_iter_skips_empty_kmers():
    m = importlib.import_module(MODULE)

    df = pd.DataFrame({
        "Classified": ["C"],
        "ID": ["s1"],
        "Length": [100],
        "Kmers": [None],
        "Taxonomy": ["60"],
    })

    out = list(m.process_chunk_iter(df))

    assert out == []


def test_process_chunk_iter_basic_regular_mode(monkeypatch):
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    monkeypatch.setattr(m, "normalize_taxid", lambda x: int(str(x).strip("()")))
    monkeypatch.setattr(m, "get_lineage_path", lambda tid: [1, tid])
    monkeypatch.setattr(
        m,
        "lineage_to_rank_map",
        lambda lineage, ranks: {r: lineage[-1] for r in ranks},
    )
    monkeypatch.setattr(
        m,
        "compute_bin_features",
        lambda counts, lineage, ranks: [len(counts)] * 28,
    )

    globals_mod._shared_lineage_map = {
        60: [1, 60],
        61: [1, 61],
    }

    df = pd.DataFrame({
        "Classified": ["C"],
        "ID": ["seq1|60"],
        "Length": [100],
        "Kmers": ["60:12 61:4"],
        "Taxonomy": ["60"],
    })

    out = list(m.process_chunk_iter(df, min_tax_kmers=10, is_training=False))

    assert len(out) == 1

    row = out[0]
    assert row["seq_id"] == "seq1|60"
    assert row["taxon"] == 60
    assert row["true_taxon"] == 60
    assert len(row["bins"]) >= 1
    assert len(row["labels_per_rank"]) == len(m.CANONICAL_RANKS)


def test_process_chunk_iter_skips_when_no_candidates(monkeypatch):
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    monkeypatch.setattr(m, "normalize_taxid", lambda x: int(str(x).strip("()")))
    monkeypatch.setattr(m, "get_lineage_path", lambda tid: [1, tid])
    monkeypatch.setattr(
        m,
        "lineage_to_rank_map",
        lambda lineage, ranks: {r: lineage[-1] for r in ranks},
    )

    globals_mod._shared_lineage_map = {60: [1, 60]}

    df = pd.DataFrame({
        "Classified": ["C"],
        "ID": ["seq1|60"],
        "Length": [100],
        "Kmers": ["60:3 61:2"],
        "Taxonomy": ["60"],
    })

    out = list(m.process_chunk_iter(df, min_tax_kmers=10, is_training=False))

    assert out == []


def test_process_chunk_iter_max_bins_per_seq(monkeypatch):
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    monkeypatch.setattr(m, "normalize_taxid", lambda x: int(str(x).strip("()")))
    monkeypatch.setattr(m, "get_lineage_path", lambda tid: [1, tid])
    monkeypatch.setattr(
        m,
        "lineage_to_rank_map",
        lambda lineage, ranks: {r: lineage[-1] for r in ranks},
    )
    monkeypatch.setattr(
        m,
        "compute_bin_features",
        lambda counts, lineage, ranks: [sum(counts.values())] * 28,
    )

    globals_mod._shared_lineage_map = {60: [1, 60]}

    df = pd.DataFrame({
        "Classified": ["C"],
        "ID": ["seq1|60"],
        "Length": [5000],
        "Kmers": ["60:2500"],
        "Taxonomy": ["60"],
    })

    out = list(
        m.process_chunk_iter(
            df,
            bin_size=1000,
            min_tax_kmers=10,
            max_bins_per_seq=2,
            is_training=False,
        )
    )

    assert len(out) == 1
    assert len(out[0]["bins"]) == 2


def test_process_chunk_iter_training_topk_and_keep_taxonomy(monkeypatch):
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    monkeypatch.setattr(m, "normalize_taxid", lambda x: int(str(x).strip("()")))
    monkeypatch.setattr(m, "get_lineage_path", lambda tid: [1, tid])
    monkeypatch.setattr(
        m,
        "lineage_to_rank_map",
        lambda lineage, ranks: {r: lineage[-1] for r in ranks},
    )
    monkeypatch.setattr(
        m,
        "compute_bin_features",
        lambda counts, lineage, ranks: [1] * 28,
    )

    globals_mod._shared_lineage_map = {
        60: [1, 60],
        61: [1, 61],
        62: [1, 62],
    }

    df = pd.DataFrame({
        "Classified": ["C"],
        "ID": ["seq1|60"],
        "Length": [100],
        "Kmers": ["61:20 62:15 60:12"],
        "Taxonomy": ["60"],
    })

    out = list(
        m.process_chunk_iter(
            df,
            min_tax_kmers=10,
            topk_taxa=1,
            neg_extra=0,
            keep_taxonomy=True,
            is_training=True,
        )
    )

    taxa = [row["taxon"] for row in out]
    assert 61 in taxa
    assert 60 in taxa


"""
Tests for process_chunk_and_write
"""

def test_process_chunk_and_write_shards(monkeypatch):
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    fake_rows = [{"seq_id": f"s{i}"} for i in range(5)]

    def fake_process_chunk_iter(
        chunk,
        bin_size=1000,
        min_tax_kmers=0,
        max_bins_per_seq=None,
        is_training=False,
        topk_taxa=8,
        neg_extra=0,
        keep_taxonomy=True,
    ):
        for r in fake_rows:
            yield r

    monkeypatch.setattr(m, "process_chunk_iter", fake_process_chunk_iter)

    globals_mod._shared_shard_size = 2
    globals_mod._shared_target_length = 0
    globals_mod._shared_to_dtype = "float32"

    calls = []

    def fake_write_shards(rows, max_batch_rows, target_length, to_dtype):
        batch = list(rows)
        calls.append(batch)
        return {"rows": len(batch), "file": f"shard{len(calls)}.pt"}

    monkeypatch.setattr(m, "_write_rows_streaming_shards", fake_write_shards)

    meta = m.process_chunk_and_write(
        chunk=None,
        max_bins_per_seq=None,
        mess_true_file=None,
        mess_input_file=None,
    )

    assert [len(b) for b in calls] == [2, 2, 1]
    assert meta["rows"] == 1
    assert meta["file"] == "shard3.pt"


def test_process_chunk_and_write_returns_none_when_no_rows(monkeypatch):
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    globals_mod._shared_shard_size = 2
    globals_mod._shared_target_length = 0
    globals_mod._shared_to_dtype = "float32"

    monkeypatch.setattr(m, "process_chunk_iter", lambda *a, **k: iter(()))

    out = m.process_chunk_and_write(chunk=None)

    assert out is None


"""
Tests for process_chunk_and_write_wrapper
"""
def test_process_chunk_and_write_wrapper_regular(monkeypatch):
    m = importlib.import_module(MODULE)

    calls = {}

    def fake_process_chunk_and_write(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return {"rows": 1}

    monkeypatch.setattr(m, "process_chunk_and_write", fake_process_chunk_and_write)

    out = m.process_chunk_and_write_wrapper(
        ("chunk", 5, "true.tsv", "input.tsv", 8, 10, 4, False)
    )

    assert out == {"rows": 1}
    assert calls["args"] == ("chunk",)
    assert calls["kwargs"]["max_bins_per_seq"] == 5
    assert calls["kwargs"]["min_tax_kmers"] == 10
    assert calls["kwargs"]["is_training"] is False
    assert "mess_true_file" not in calls["kwargs"]
    assert "mess_input_file" not in calls["kwargs"]
    assert "topk_taxa" not in calls["kwargs"]
    assert "neg_extra" not in calls["kwargs"]


def test_process_chunk_and_write_wrapper_training(monkeypatch):
    m = importlib.import_module(MODULE)

    calls = {}

    def fake_process_chunk_and_write(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return {"rows": 2}

    monkeypatch.setattr(m, "process_chunk_and_write", fake_process_chunk_and_write)

    out = m.process_chunk_and_write_wrapper(
        ("chunk", 5, "true.tsv", "input.tsv", 3, 7, 2, True)
    )

    assert out == {"rows": 2}
    assert calls["args"] == ("chunk",)
    assert calls["kwargs"]["max_bins_per_seq"] == 5
    assert calls["kwargs"]["mess_true_file"] == "true.tsv"
    assert calls["kwargs"]["mess_input_file"] == "input.tsv"
    assert calls["kwargs"]["topk_taxa"] == 3
    assert calls["kwargs"]["min_tax_kmers"] == 7
    assert calls["kwargs"]["neg_extra"] == 2
    assert calls["kwargs"]["is_training"] is True
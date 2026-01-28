import importlib
import pandas as pd

MODULE = "perseus.features.processing"
GLOBALS_MODULE = "perseus.utils.globals"

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

    # regex only matches \d+:\d+, so junk should be skipped
    s = "60:5 junk 61:x 10:1 foo:bar 999:7"
    toks = list(m.iter_kmer_tokens(s))
    assert toks == [(60, 5), (10, 1), (999, 7)]


def test_iter_kmer_tokens_empty_and_none():
    m = importlib.import_module(MODULE)

    assert list(m.iter_kmer_tokens("")) == []
    assert list(m.iter_kmer_tokens(None)) == []
    assert list(m.iter_kmer_tokens(123)) == []  # non-str input

""" 
Tests for extract_tax_counts and extract_tax_context_chunk
"""
def test_extract_tax_counts_basic(monkeypatch):
    m = importlib.import_module(MODULE)

    # Make normalize_taxid trivial so we don't hit NCBI
    m.normalize_taxid.cache_clear()
    monkeypatch.setattr(m, "normalize_taxid", lambda x: int(x))

    kmer_list = ["60:5", "61:3", "60:2"]  # 60 appears twice
    out = m.extract_tax_counts(kmer_list)

    assert out[60] == 7     # 5 + 2
    assert out[61] == 3
    assert len(out) == 2


def test_extract_tax_counts_with_bad_tokens(monkeypatch):
    m = importlib.import_module(MODULE)

    # identity normalize
    monkeypatch.setattr(m, "normalize_taxid", lambda x: int(x))

    kmer_list = ["60:5", "badtoken", "61:x", "abc:2"]
    out = m.extract_tax_counts(kmer_list)

    # Only the valid token should be parsed
    assert out == {60: 5}


def test_extract_tax_counts_empty(monkeypatch):
    m = importlib.import_module(MODULE)

    monkeypatch.setattr(m, "normalize_taxid", lambda x: int(x))

    out = m.extract_tax_counts([])
    assert out == {}      # no entries


def test_extract_tax_counts_normalization(monkeypatch):
    m = importlib.import_module(MODULE)

    # A normalization function that “collapses” species to genus for testing:
    def fake_norm(x):
        return 50 if int(x) in (60, 61) else int(x)

    monkeypatch.setattr(m, "normalize_taxid", fake_norm)

    kmer_list = ["60:5", "61:3"]   # both should normalize to 50
    out = m.extract_tax_counts(kmer_list)

    assert out == {50: 8}


def test_extract_tax_context_chunk_basic(monkeypatch):
    m = importlib.import_module(MODULE)

    # Monkeypatch normalize_taxid to avoid filesystem / NCBI calls.
    # We pretend normalize_taxid just returns the input as int.
    m.normalize_taxid.cache_clear()
    monkeypatch.setattr(m, "normalize_taxid", lambda x: int(x))

    # Create a fake chunk with:
    #   - valid kmers
    #   - duplicate kmers
    #   - a row with NaN kmers (should be dropped)
    df = pd.DataFrame({
        "ID": ["seq1", "seq2", "seq3"],
        "Taxonomy": ["(60)", "(61)", "(50)"],
        "Kmers": [
            "60:5 61:3",      # seq1
            "60:2 60:3 10:1", # seq2 has repeated 60
            None              # seq3 should be dropped
        ]
    })

    out = m.extract_tax_context_chunk(df)

    # seq1: two taxids
    assert "seq1" in out
    assert out["seq1"] == {60: 5, 61: 3}

    # seq2: repeated 60:2 and 60:3 → sum to 5
    assert "seq2" in out
    assert out["seq2"] == {60: 5, 10: 1}

    # seq3 should not appear (NaN Kmers)
    assert "seq3" not in out


def test_add_to_bins_distribution():
    m = importlib.import_module(MODULE)
    bins = {}
    # bin_size=10; add count=25 starting at pos=0 -> bins 0:[10],1:[10],2:[5]
    pos = m.add_to_bins(bins, bin_size=10, taxid=60, count=25, cur_pos=0)
    assert pos == 25
    assert bins[0][60] == 10
    assert bins[1][60] == 10
    assert bins[2][60] == 5


"""
Tests for process_chunk_and_write
"""    
def test_process_chunk_and_write_shards(monkeypatch):
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    # 5 fake rows so we can test multiple flushes
    fake_rows = [
        {"seq_id": f"s{i}"} for i in range(5)
    ]

    def fake_process_chunk_iter(
        chunk, 
        bin_size=1000, 
        min_tax_kmers=0, 
        max_bins_per_seq=None,
        is_training=False
    ):
        for r in fake_rows:
            yield r

    monkeypatch.setattr(m, "process_chunk_iter", fake_process_chunk_iter)

    # Shard mode: flush every _shared_shard_size rows
    globals_mod._shared_shard_size = 2
    globals_mod._shared_target_length = 0
    globals_mod._shared_to_dtype = "float32"

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

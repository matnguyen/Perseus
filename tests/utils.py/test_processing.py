import importlib
import numpy as np
import pandas as pd
import pytest

MODULE = "taxoncnn.utils.processing"

"""
Tests for parse_kmers and iter_kmer_tokens
"""
def test_parse_kmers_basic():
    import pandas as pd

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

def test_resample():
    m = importlib.import_module(MODULE)
    x = np.arange(6, dtype=np.float32).reshape(3,2)   # T=3,C=2
    y = m._resample_TN_to_T(x, 5)
    assert y.shape == (5,2)
    # monotonic per-column
    assert np.all(np.diff(y[:,0]) >= 0)

"""
Tests for canonicalize_rank
"""
@pytest.mark.parametrize("raw,expected", [
    ("Kingdom", "superkingdom"),     # special-case mapping
    ("kingdom", "superkingdom"),
    ("superkingdom", "superkingdom"),
    ("phylum", "phylum"),
    ("Class", "class"),
    ("order", "order"),
    ("family", "family"),
    ("genus", "genus"),
    ("species", "species"),
    ("strain", "strain"),
    ("subclass", "class"),    # TODO: decide on handling sub-canonical ranks
    ("weird_rank", None),
    ("", None),
    (None, None),
])
def test_canonicalize_rank_cases(raw, expected):
    m = importlib.import_module(MODULE)
    assert m.canonicalize_rank(raw) == expected

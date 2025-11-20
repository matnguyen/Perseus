import importlib
import numpy as np
import pytest

MODULE = "taxoncnn.features.features"
GLOBAL_MODULE = "taxoncnn.utils.globals"
CONSTANTS_MODULE = "taxoncnn.utils.constants"

"""
Tests for compute_bin_features
""" 
def test_compute_bin_features_only_pred_tax():
    """
    All kmers come from the predicted species (60). We check:
      - raw_total fraction is 1.0 (all kmers at pred taxid)
      - at species rank: fi = 1.0, fo = fd = 0
      - at genus rank: all kmers are descendants (fd = 1.0, fi = fo = 0)
    """
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBAL_MODULE)
    constants_mod = importlib.import_module(CONSTANTS_MODULE)

    # bin has only species 60
    kmer_tax_counts = {60: 10}
    pred_lineage = globals_mod._shared_lineage_map[60]  # from FakeNCBI in conftest

    vec = m.compute_bin_features(kmer_tax_counts, pred_lineage, constants_mod.CANONICAL_RANKS)

    # length = 1 (raw_total) + 3 * len(CANONICAL_RANKS)
    assert len(vec) == 1 + 3 * len(constants_mod.CANONICAL_RANKS)

    # raw_total is fraction of kmers at exactly the predicted node
    assert pytest.approx(vec[0], rel=1e-6) == 1.0

    # species rank index
    sp_ix = constants_mod.RANK_INDEX["species"]
    sp_start = 1 + 3 * sp_ix
    fi_sp, fo_sp, fd_sp = vec[sp_start], vec[sp_start + 1], vec[sp_start + 2]

    # at species rank, all kmers are at the predicted species node
    assert pytest.approx(fi_sp, rel=1e-6) == 1.0   # in_lineage
    assert pytest.approx(fo_sp, rel=1e-6) == 0.0   # out_of_lineage
    assert pytest.approx(fd_sp, rel=1e-6) == 0.0   # descendants

    # genus rank index
    g_ix = constants_mod.RANK_INDEX["genus"]
    g_start = 1 + 3 * g_ix
    fi_g, fo_g, fd_g = vec[g_start], vec[g_start + 1], vec[g_start + 2]

    # at genus rank, species 60 is a descendant of genus(50),
    # so all kmers are counted as descendants for that rank
    assert pytest.approx(fi_g, rel=1e-6) == 0.0
    assert pytest.approx(fo_g, rel=1e-6) == 0.0
    assert pytest.approx(fd_g, rel=1e-6) == 1.0


def test_compute_bin_features_pred_vs_sibling_species():
    """
    Bin contains kmers from predicted species 60 and sibling species 61.
    We check:
      - raw_total = kmer fraction from the predicted taxid itself (60)
      - at species rank: 60 is in-lineage, 61 is out-of-lineage
      - at genus rank: both are descendants of genus(50)
    """
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBAL_MODULE)
    constants_mod = importlib.import_module(CONSTANTS_MODULE)

    # 5 kmers from 60, 3 from sibling 61
    kmer_tax_counts = {60: 5, 61: 3}
    total = 8.0
    pred_lineage = globals_mod._shared_lineage_map[60]

    vec = m.compute_bin_features(kmer_tax_counts, pred_lineage, constants_mod.CANONICAL_RANKS)

    # raw_total = fraction of kmers exactly at predicted taxid
    assert pytest.approx(vec[0], rel=1e-6) == 5.0 / total

    # ---- species rank ----
    sp_ix = constants_mod.RANK_INDEX["species"]
    sp_start = 1 + 3 * sp_ix
    fi_sp, fo_sp, fd_sp = vec[sp_start], vec[sp_start + 1], vec[sp_start + 2]

    # species(60) is exactly the predicted node → in_lineage
    # sibling species(61) is outside that species-level lineage → out_of_lineage
    assert pytest.approx(fi_sp, rel=1e-6) == 5.0 / total   # 60
    assert pytest.approx(fo_sp, rel=1e-6) == 3.0 / total   # 61
    assert pytest.approx(fd_sp, rel=1e-6) == 0.0

    # ---- genus rank ----
    g_ix = constants_mod.RANK_INDEX["genus"]
    g_start = 1 + 3 * g_ix
    fi_g, fo_g, fd_g = vec[g_start], vec[g_start + 1], vec[g_start + 2]

    # at genus rank, both 60 and 61 are descendants of genus(50),
    # not sitting on the genus node itself, so everything is counted as "descendant"
    assert pytest.approx(fi_g, rel=1e-6) == 0.0
    assert pytest.approx(fo_g, rel=1e-6) == 0.0
    assert pytest.approx(fd_g, rel=1e-6) == 1.0


def test_compute_bin_features_empty_bin():
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBAL_MODULE)
    constants_mod = importlib.import_module(CONSTANTS_MODULE)

    pred_lineage = globals_mod._shared_lineage_map[60]
    vec = m.compute_bin_features({}, pred_lineage, constants_mod.CANONICAL_RANKS)

    # all zeros if no kmers
    assert all(float(v) == 0.0 for v in vec)
    assert len(vec) == 1 + 3 * len(constants_mod.CANONICAL_RANKS)
    
"""
Tests for _resample_TN_to_T
"""
def test_resample():
    m = importlib.import_module(MODULE)
    x = np.arange(6, dtype=np.float32).reshape(3,2)   # T=3,C=2
    y = m._resample_TN_to_T(x, 5)
    assert y.shape == (5,2)
    # monotonic per-column
    assert np.all(np.diff(y[:,0]) >= 0)
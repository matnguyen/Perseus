# tests/utils/test_select_one_row_per_seq.py

import importlib
import numpy as np
import pandas as pd
import pytest

MODULE = "perseus.utils.filter_utils"


class DummyBar:
    def __enter__(self):
        return lambda *args, **kwargs: None

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def m(monkeypatch):
    mod = importlib.import_module(MODULE)
    monkeypatch.setattr(mod, "alive_bar", lambda *args, **kwargs: DummyBar())
    return mod


def test_raises_when_no_prob_columns(m):
    df = pd.DataFrame({
        "sequence_id": ["seq1", "seq1"],
        "x": [1, 2],
    })

    with pytest.raises(ValueError, match="No prob_\\* rank columns found"):
        m.select_one_row_per_seq(df)


def test_raises_on_bad_threshold_type(m):
    df = pd.DataFrame({
        "sequence_id": ["seq1"],
        "prob_species": [0.9],
    })

    with pytest.raises(TypeError, match="thresholds must be a float or dict"):
        m.select_one_row_per_seq(df, thresholds=[0.5])


def test_infers_ranks_from_columns(m):
    df = pd.DataFrame({
        "sequence_id": ["seq1", "seq1"],
        "taxid": [10, 20],
        "prob_genus": [0.4, 0.9],
        "prob_species": [0.8, 0.1],
    })

    out = m.select_one_row_per_seq(df, thresholds=0.5)

    assert len(out) == 1
    row = out.iloc[0]

    # deepest passing rank is species because max species = 0.8 >= 0.5
    assert row["taxid"] == 10
    assert row["chosen_rank"] == "species"
    assert row["chosen_rank_ix"] == 1
    assert row["chosen_prob_at_rank"] == pytest.approx(0.8)


def test_deepest_passing_rank_is_used_not_highest_overall_prob(m):
    df = pd.DataFrame({
        "sequence_id": ["seq1", "seq1"],
        "taxid": [111, 222],
        "prob_genus":   [0.95, 0.60],
        "prob_species": [0.40, 0.70],
    })

    out = m.select_one_row_per_seq(df, thresholds=0.5)
    row = out.iloc[0]

    # genus passes for both, species passes only for taxid 222
    # deepest passing rank is species, so choose row 222
    assert row["taxid"] == 222
    assert row["chosen_rank"] == "species"
    assert row["chosen_prob_at_rank"] == pytest.approx(0.70)


def test_fallback_to_deepest_finite_rank_when_nothing_passes_threshold(m):
    df = pd.DataFrame({
        "sequence_id": ["seq1", "seq1"],
        "taxid": [1, 2],
        "prob_genus":   [0.20, 0.30],
        "prob_species": [0.10, 0.40],
    })

    out = m.select_one_row_per_seq(df, thresholds=0.95)
    row = out.iloc[0]

    # no rank passes threshold, so fallback to deepest finite rank = species
    assert row["taxid"] == 2
    assert row["chosen_rank"] == "species"
    assert row["chosen_prob_at_rank"] == pytest.approx(0.40)


def test_all_nan_rows_fallback_to_first_original_row(m):
    df = pd.DataFrame({
        "sequence_id": ["seq1", "seq1"],
        "taxid": [100, 200],
        "prob_genus": [np.nan, np.nan],
        "prob_species": [np.nan, np.nan],
    })

    out = m.select_one_row_per_seq(df, thresholds=0.5)
    row = out.iloc[0]

    assert row["taxid"] == 100
    assert row["chosen_rank_ix"] == -1
    assert pd.isna(row["chosen_prob_at_rank"])
    assert row["chosen_rank"] is None


def test_nonfinite_values_do_not_force_fallback_when_shallower_rank_passes(m):
    df = pd.DataFrame({
        "sequence_id": ["seq1", "seq1"],
        "taxid": [1, 2],
        "prob_genus":   [0.9, 0.9],
        "prob_species": [np.nan, 0.1],
    })

    out = m.select_one_row_per_seq(
        df,
        thresholds={"genus": 0.5, "species": 0.95},
        tie_breaker="sum_to_rank",
    )
    row = out.iloc[0]

    assert row["taxid"] == 1
    assert row["chosen_rank"] == "genus"


def test_nonfinite_values_are_ignored_in_fallback_to_deepest_finite_rank(m):
    df = pd.DataFrame({
        "sequence_id": ["seq1", "seq1"],
        "taxid": [1, 2],
        "prob_genus":   [0.9, 0.9],
        "prob_species": [np.nan, 0.1],
    })

    out = m.select_one_row_per_seq(
        df,
        thresholds={"genus": 0.95, "species": 0.95},
        tie_breaker="sum_to_rank",
    )
    row = out.iloc[0]

    assert row["taxid"] == 2
    assert row["chosen_rank"] == "species"


def test_prefer_lineage_restricts_when_lineage_rows_exist(m):
    df = pd.DataFrame({
        "sequence_id": ["seq1", "seq1"],
        "taxid": [1, 2],
        "perseus_in_lineage": [False, True],
        "prob_genus":   [0.99, 0.60],
        "prob_species": [0.99, 0.55],
    })

    out = m.select_one_row_per_seq(
        df,
        thresholds=0.5,
        lineage_filter_col="perseus_in_lineage",
        prefer_lineage=True,
    )
    row = out.iloc[0]

    # although taxid 1 scores better, taxid 2 is the only lineage row
    assert row["taxid"] == 2
    assert row["chosen_rank"] == "species"


def test_prefer_lineage_falls_back_to_all_rows_if_no_lineage_rows(m):
    df = pd.DataFrame({
        "sequence_id": ["seq1", "seq1"],
        "taxid": [1, 2],
        "perseus_in_lineage": [False, False],
        "prob_genus":   [0.60, 0.70],
        "prob_species": [0.80, 0.40],
    })

    out = m.select_one_row_per_seq(
        df,
        thresholds=0.5,
        lineage_filter_col="perseus_in_lineage",
        prefer_lineage=True,
    )
    row = out.iloc[0]

    # no lineage rows exist, so use all rows; species is deepest passing rank
    assert row["taxid"] == 1
    assert row["chosen_rank"] == "species"


def test_prefer_lineage_false_does_not_filter(m):
    df = pd.DataFrame({
        "sequence_id": ["seq1", "seq1"],
        "taxid": [1, 2],
        "perseus_in_lineage": [True, False],
        "prob_genus":   [0.5, 0.9],
        "prob_species": [0.5, 0.95],
    })

    out = m.select_one_row_per_seq(
        df,
        thresholds=0.5,
        lineage_filter_col="perseus_in_lineage",
        prefer_lineage=False,
    )
    row = out.iloc[0]

    assert row["taxid"] == 2
    assert row["chosen_rank"] == "species"


def test_tie_breaker_p_only_picks_first_tied_row(m):
    df = pd.DataFrame({
        "sequence_id": ["seq1", "seq1"],
        "taxid": [10, 20],
        "prob_genus":   [0.90, 0.80],
        "prob_species": [0.70, 0.70],
    })

    out = m.select_one_row_per_seq(df, thresholds=0.5, tie_breaker="p_only")
    row = out.iloc[0]

    # tied on chosen rank species; p_only keeps first tied row
    assert row["taxid"] == 10
    assert row["chosen_rank"] == "species"


def test_tie_breaker_sum_to_rank_breaks_species_tie(m):
    df = pd.DataFrame({
        "sequence_id": ["seq1", "seq1"],
        "taxid": [10, 20],
        "prob_genus":   [0.60, 0.90],
        "prob_species": [0.70, 0.70],
    })

    out = m.select_one_row_per_seq(df, thresholds=0.5, tie_breaker="sum_to_rank")
    row = out.iloc[0]

    # species tied at 0.70; sum to species favors taxid 20
    assert row["taxid"] == 20
    assert row["chosen_rank"] == "species"


def test_tie_breaker_deep_then_sum_uses_mass_below_rank(m):
    df = pd.DataFrame({
        "sequence_id": ["seq1", "seq1"],
        "taxid": [10, 20],
        "prob_genus":      [0.90, 0.90],
        "prob_species":    [0.50, 0.50],
        "prob_subspecies": [0.10, 0.30],
    })

    out = m.select_one_row_per_seq(
        df,
        thresholds={"genus": 0.5, "species": 0.5, "subspecies": 0.95},
        tie_breaker="deep_then_sum",
    )
    row = out.iloc[0]

    # chosen rank = species
    # tied on species and tied on sum_to_species, so mass below species decides
    assert row["taxid"] == 20
    assert row["chosen_rank"] == "species"


def test_multiple_sequences_returns_one_per_sequence_preserving_group_order(m):
    df = pd.DataFrame({
        "sequence_id": ["seqB", "seqA", "seqB", "seqA"],
        "taxid": [1, 2, 3, 4],
        "prob_species": [0.1, 0.9, 0.8, 0.2],
    })

    out = m.select_one_row_per_seq(df, thresholds=0.5)

    assert list(out["sequence_id"]) == ["seqB", "seqA"]
    assert list(out["taxid"]) == [3, 2]


def test_custom_sequence_column(m):
    df = pd.DataFrame({
        "contig": ["c1", "c1", "c2", "c2"],
        "taxid": [1, 2, 3, 4],
        "prob_species": [0.1, 0.8, 0.7, 0.2],
    })

    out = m.select_one_row_per_seq(
        df,
        sequence_col="contig",
        thresholds=0.5,
    )

    assert list(out["contig"]) == ["c1", "c2"]
    assert list(out["taxid"]) == [2, 3]


def test_threshold_dict_defaults_missing_ranks_to_point_five(m):
    df = pd.DataFrame({
        "sequence_id": ["seq1", "seq1"],
        "taxid": [1, 2],
        "prob_genus":   [0.40, 0.60],
        "prob_species": [0.45, 0.49],
    })

    out = m.select_one_row_per_seq(
        df,
        thresholds={"species": 0.9},  # genus should default to 0.5
    )
    row = out.iloc[0]

    # species doesn't pass; genus does for taxid 2
    assert row["taxid"] == 2
    assert row["chosen_rank"] == "genus"
    assert row["chosen_prob_at_rank"] == pytest.approx(0.60)


def test_nondefault_index_and_lineage_filter_work(m):
    df = pd.DataFrame({
        "sequence_id": ["seq1", "seq1"],
        "taxid": [1, 2],
        "perseus_in_lineage": [False, True],
        "prob_species": [0.99, 0.60],
    }, index=[100, 200])

    out = m.select_one_row_per_seq(
        df,
        thresholds=0.5,
        lineage_filter_col="perseus_in_lineage",
        prefer_lineage=True,
    )

    assert out.iloc[0]["taxid"] == 2
    assert list(out.index) == [200]
import importlib
import numpy as np
import pytest

MODULE = "perseus.trainer.metrics"


@pytest.mark.dev
def test_auroc_perfect():
    m = importlib.import_module(MODULE)
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.8, 0.9])

    auroc = m.binary_auroc(y_true, y_score)

    assert np.isclose(auroc, 1.0)


@pytest.mark.dev
def test_auroc_random():
    m = importlib.import_module(MODULE)
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])

    auroc = m.binary_auroc(y_true, y_score)

    assert 0.0 <= auroc <= 1.0


@pytest.mark.dev
def test_auroc_all_zeros():
    m = importlib.import_module(MODULE)
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])

    auroc = m.binary_auroc(y_true, y_score)

    assert auroc == 0.5


@pytest.mark.dev
def test_auroc_all_ones():
    m = importlib.import_module(MODULE)
    y_true = np.array([1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])

    auroc = m.binary_auroc(y_true, y_score)

    assert auroc == 0.5


@pytest.mark.dev
def test_auroc_inverted():
    m = importlib.import_module(MODULE)
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.9, 0.8, 0.4, 0.1])

    auroc = m.binary_auroc(y_true, y_score)

    assert np.isclose(auroc, 0.0)


@pytest.mark.dev
def test_precision_recall_curve_basic():
    m = importlib.import_module(MODULE)
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8])

    precisions, recalls = m.precision_recall_curve_from_scores(y_true, y_score)

    assert len(precisions) == len(y_true) + 1
    assert len(recalls) == len(y_true) + 1
    assert precisions[0] == 1.0
    assert recalls[0] == 0.0
    assert np.isclose(recalls[-1], 1.0)


@pytest.mark.dev
def test_precision_recall_curve_all_negative_labels():
    m = importlib.import_module(MODULE)
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.4, 0.3, 0.2, 0.1])

    precisions, recalls = m.precision_recall_curve_from_scores(y_true, y_score)

    assert len(precisions) == 5
    assert len(recalls) == 5
    assert np.allclose(recalls, 0.0)
    assert precisions[0] == 1.0


@pytest.mark.dev
def test_binary_aupr_perfect():
    m = importlib.import_module(MODULE)
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    aupr = m.binary_aupr(y_true, y_score)

    assert np.isclose(aupr, 1.0)


@pytest.mark.dev
def test_binary_aupr_inverted_is_low():
    m = importlib.import_module(MODULE)
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])

    aupr = m.binary_aupr(y_true, y_score)

    assert 0.0 <= aupr < 0.5


@pytest.mark.dev
def test_confusion_matrix_from_threshold_basic():
    m = importlib.import_module(MODULE)
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.2, 0.8, 0.1])

    tp, fp, fn, tn = m.confusion_matrix_from_threshold(y_true, y_score, thr=0.5)

    assert tp == 1
    assert fp == 1
    assert fn == 1
    assert tn == 1


@pytest.mark.dev
def test_confusion_matrix_threshold_is_strict_greater_than():
    m = importlib.import_module(MODULE)
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.5, 0.5, 0.6, 0.4])

    tp, fp, fn, tn = m.confusion_matrix_from_threshold(y_true, y_score, thr=0.5)

    assert tp == 1
    assert fp == 0
    assert fn == 1
    assert tn == 2


@pytest.mark.dev
def test_f1_from_counts_basic():
    m = importlib.import_module(MODULE)

    f1, prec, rec = m.f1_from_counts(tp=4, fp=1, fn=3)

    assert np.isclose(prec, 4 / 5)
    assert np.isclose(rec, 4 / 7)
    assert np.isclose(f1, 2 * (4 / 5) * (4 / 7) / ((4 / 5) + (4 / 7)))


@pytest.mark.dev
def test_f1_from_counts_zero_case():
    m = importlib.import_module(MODULE)

    f1, prec, rec = m.f1_from_counts(tp=0, fp=0, fn=5)

    assert f1 == 0.0
    assert prec == 0.0
    assert rec == 0.0


@pytest.mark.dev
def test_f1_from_counts_perfect():
    m = importlib.import_module(MODULE)

    f1, prec, rec = m.f1_from_counts(tp=5, fp=0, fn=0)

    assert f1 == 1.0
    assert prec == 1.0
    assert rec == 1.0
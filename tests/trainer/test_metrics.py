import importlib
import numpy as np

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
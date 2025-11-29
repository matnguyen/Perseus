
import numpy as np

def binary_auroc(y_true, y_score):
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (AUROC) for binary classification

    Args:
        y_true (np.ndarray): Ground truth binary labels (0 or 1), shape (N,)
        y_score (np.ndarray): Predicted scores or probabilities, shape (N,)

    Returns:
        float: AUROC value in [0, 1], or 0.5 if only one class is present
    """
    y_true = y_true.astype(np.int32)
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    p = y.sum()
    n = len(y) - p
    if p == 0 or n == 0:
        return 0.5
    tp = 0; fp = 0
    tps = [0]; fps = [0]
    for yi in y:
        if yi == 1: tp += 1
        else: fp += 1
        tps.append(tp); fps.append(fp)
    tps = np.array(tps, dtype=np.float64); fps = np.array(fps, dtype=np.float64)
    tpr = tps / p; fpr = fps / n
    return np.trapz(tpr, fpr)


def precision_recall_curve_from_scores(y_true, y_score):
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order].astype(np.int32)
    tp = 0; fp = 0
    tps = [0]; fps = [0]
    for yi in y:
        if yi == 1: tp += 1
        else: fp += 1
        tps.append(tp); fps.append(fp)
    tps = np.array(tps, dtype=np.float64)
    fps = np.array(fps, dtype=np.float64)
    P = y_true.sum()
    recalls = tps / max(P, 1.0)
    precisions = tps / np.maximum(tps + fps, 1.0)
    recalls[0] = 0.0
    precisions[0] = 1.0
    return precisions, recalls


def binary_aupr(y_true, y_score):
    p, r = precision_recall_curve_from_scores(y_true, y_score)
    return float(np.trapz(p, r))

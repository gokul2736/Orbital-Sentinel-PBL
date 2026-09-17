"""Regression and threshold-based evaluation metrics."""

from typing import Optional

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    max_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def _safe_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Core regression metrics for log10(collision probability) predictions."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "max_error": float(max_error(y_true, y_pred)),
        "correlation": _safe_correlation(y_true, y_pred),
    }


RISK_BANDS = {
    "floor": (-np.inf, -29.99),
    "low": (-29.99, -15.0),
    "medium": (-15.0, -7.0),
    "high": (-7.0, np.inf),
}


def evaluate_by_risk_band(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-band regression metrics. Bands: floor (-30), low (-30 to -15), medium (-15 to -7), high (>-7)."""
    results = {}
    for band_name, (lo, hi) in RISK_BANDS.items():
        mask = (y_true > lo) & (y_true <= hi)
        n = int(mask.sum())
        if n == 0:
            results[band_name] = {"n": 0}
            continue
        metrics = evaluate_regression(y_true[mask], y_pred[mask])
        metrics["n"] = n
        results[band_name] = metrics
    return results


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = -5.0,
) -> dict:
    """Binary classification metrics treating predictions above threshold as positive (high-risk)."""
    y_true_bin = (y_true > threshold).astype(int)
    y_pred_bin = (y_pred > threshold).astype(int)

    result = {
        "threshold": threshold,
        "n_true_positive": int(y_true_bin.sum()),
        "n_pred_positive": int(y_pred_bin.sum()),
        "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
    }

    if len(np.unique(y_true_bin)) == 2:
        result["auc"] = float(roc_auc_score(y_true_bin, y_pred))
    else:
        result["auc"] = float("nan")

    return result

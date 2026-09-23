"""Optimize binary classification threshold for operational decisions."""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def optimize_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "f1",
    search_range: tuple[float, float] = (-8.0, -3.0),
    n_steps: int = 100,
) -> dict:
    """Find the optimal binary classification threshold.

    Args:
        y_true: True risk values (log10 Pc)
        y_pred: Predicted risk values
        metric: Optimization target — "f1", "recall", "precision", or "f2"
        search_range: Range of thresholds to search
        n_steps: Number of threshold values to evaluate
    """
    thresholds = np.linspace(search_range[0], search_range[1], n_steps)
    results = []

    for thresh in thresholds:
        y_true_bin = (y_true > thresh).astype(int)
        y_pred_bin = (y_pred > thresh).astype(int)

        if y_true_bin.sum() == 0 or y_pred_bin.sum() == 0:
            continue

        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        beta = 2.0
        f2 = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec) if (prec + rec) > 0 else 0.0

        results.append({
            "threshold": float(thresh),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "f2": float(f2),
            "n_true_pos": int(y_true_bin.sum()),
            "n_pred_pos": int(y_pred_bin.sum()),
        })

    if not results:
        return {"optimal_threshold": -5.0, "optimal_score": 0.0, "metric": metric, "results": []}

    best = max(results, key=lambda r: r[metric])

    return {
        "optimal_threshold": best["threshold"],
        "optimal_score": best[metric],
        "metric": metric,
        "best_metrics": best,
        "all_results": results,
    }


def multi_threshold_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: list[float] = None,
) -> list[dict]:
    """Evaluate binary classification at multiple operational thresholds."""
    if thresholds is None:
        thresholds = [-3.0, -4.0, -5.0, -6.0, -7.0]

    results = []
    for thresh in thresholds:
        y_true_bin = (y_true > thresh).astype(int)
        y_pred_bin = (y_pred > thresh).astype(int)

        entry = {
            "threshold": thresh,
            "n_positive": int(y_true_bin.sum()),
            "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
            "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
            "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        }

        if len(np.unique(y_true_bin)) == 2:
            entry["auc"] = float(roc_auc_score(y_true_bin, y_pred))
        else:
            entry["auc"] = None

        results.append(entry)

    return results

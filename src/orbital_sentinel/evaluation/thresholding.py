"""Operational threshold selection and risk categorization."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


RISK_CATEGORIES = {
    "NEGLIGIBLE": (-np.inf, -20.0),
    "LOW": (-20.0, -10.0),
    "MODERATE": (-10.0, -5.0),
    "HIGH": (-5.0, -3.0),
    "CRITICAL": (-3.0, np.inf),
}


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "f1",
    search_range: tuple = (-10.0, -1.0),
    n_steps: int = 200,
) -> float:
    """Find the threshold on y_pred that maximizes the given binary metric for high-risk detection."""
    thresholds = np.linspace(search_range[0], search_range[1], n_steps)
    metric_fn = {"f1": f1_score, "precision": precision_score, "recall": recall_score}[
        metric
    ]

    best_score = -1.0
    best_thresh = thresholds[0]

    for t in thresholds:
        y_true_bin = (y_true > t).astype(int)
        y_pred_bin = (y_pred > t).astype(int)
        if y_pred_bin.sum() == 0 or y_true_bin.sum() == 0:
            continue
        score = metric_fn(y_true_bin, y_pred_bin, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = t

    return float(best_thresh)


def apply_risk_categories(predictions: np.ndarray) -> np.ndarray:
    """Map continuous log10(probability) predictions to categorical risk labels."""
    categories = np.empty(len(predictions), dtype=object)
    for label, (lo, hi) in RISK_CATEGORIES.items():
        mask = (predictions > lo) & (predictions <= hi)
        categories[mask] = label
    return categories

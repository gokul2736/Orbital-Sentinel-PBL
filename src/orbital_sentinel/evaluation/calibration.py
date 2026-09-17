"""Calibration analysis for regression prediction intervals."""

from typing import List, Optional

import numpy as np


def calibration_curve(
    y_true: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    confidence_levels: Optional[List[float]] = None,
) -> dict:
    """Evaluate how well prediction intervals are calibrated.

    For each target confidence level, computes the actual coverage
    (fraction of y_true that falls within [lower, upper]).
    """
    if confidence_levels is None:
        confidence_levels = [0.5, 0.7, 0.9, 0.95]

    y_true = np.asarray(y_true)
    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)

    expected_coverage = np.array(confidence_levels, dtype=float)
    actual_coverage = np.array(
        [
            float(np.mean((y_true >= lower_bounds) & (y_true <= upper_bounds)))
            for _ in confidence_levels
        ],
        dtype=float,
    )

    within = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    n = len(y_true)

    sorted_idx = np.argsort(confidence_levels)
    sorted_levels = expected_coverage[sorted_idx]

    actual_coverages = []
    for level in sorted_levels:
        alpha = 1.0 - level
        center = (lower_bounds + upper_bounds) / 2.0
        half_width = (upper_bounds - lower_bounds) / 2.0

        if np.all(half_width == half_width[0]):
            coverage = float(np.mean(within))
        else:
            widths = upper_bounds - lower_bounds
            width_order = np.argsort(widths)
            target_count = int(np.ceil(level * n))
            target_count = min(target_count, n)

            selected = width_order[:target_count]
            coverage = float(np.mean(within[selected])) if len(selected) > 0 else 0.0

        actual_coverages.append(coverage)

    reorder = np.argsort(sorted_idx)
    actual_coverage = np.array(actual_coverages)[reorder]

    calibration_error = float(np.mean(np.abs(expected_coverage - actual_coverage)))

    return {
        "expected_coverage": expected_coverage.tolist(),
        "actual_coverage": actual_coverage.tolist(),
        "calibration_error": calibration_error,
    }


def reliability_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Bin predictions into quantiles and compare against actuals."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    try:
        bin_edges = np.unique(np.percentile(y_pred, np.linspace(0, 100, n_bins + 1)))
    except IndexError:
        bin_edges = np.array([y_pred.min(), y_pred.max()])

    bin_indices = np.digitize(y_pred, bin_edges[1:-1])

    bins = []
    total_weighted_error = 0.0
    total_count = 0

    for i in range(len(bin_edges) - 1):
        mask = bin_indices == i
        count = int(mask.sum())
        if count == 0:
            continue

        mean_predicted = float(np.mean(y_pred[mask]))
        mean_actual = float(np.mean(y_true[mask]))
        mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

        bins.append({
            "bin": i,
            "lower": float(bin_edges[i]),
            "upper": float(bin_edges[i + 1]),
            "count": count,
            "mean_predicted": mean_predicted,
            "mean_actual": mean_actual,
            "mae": mae,
        })

        total_weighted_error += mae * count
        total_count += count

    reliability_score = 1.0 - (total_weighted_error / total_count) / (
        np.std(y_true) + 1e-10
    ) if total_count > 0 else 0.0
    reliability_score = float(np.clip(reliability_score, 0.0, 1.0))

    return {
        "bins": bins,
        "reliability_score": reliability_score,
    }

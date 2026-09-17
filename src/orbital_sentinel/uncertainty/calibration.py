"""Calibration analysis for prediction intervals."""

import numpy as np


def calibration_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    intervals_lower: np.ndarray,
    intervals_upper: np.ndarray,
) -> dict:
    """Assess how well prediction intervals capture true values.

    Returns coverage (fraction of true values within intervals),
    mean interval width, and per-band coverage breakdown.
    """
    in_interval = (y_true >= intervals_lower) & (y_true <= intervals_upper)
    widths = intervals_upper - intervals_lower

    result = {
        "coverage": float(in_interval.mean()),
        "mean_width": float(widths.mean()),
        "median_width": float(np.median(widths)),
        "n_samples": int(len(y_true)),
    }

    bands = {
        "floor": (-np.inf, -29.99),
        "low": (-29.99, -15.0),
        "medium": (-15.0, -7.0),
        "high": (-7.0, np.inf),
    }
    band_coverage = {}
    for name, (lo, hi) in bands.items():
        mask = (y_true > lo) & (y_true <= hi)
        n = int(mask.sum())
        if n == 0:
            band_coverage[name] = {"n": 0, "coverage": float("nan")}
        else:
            band_coverage[name] = {
                "n": n,
                "coverage": float(in_interval[mask].mean()),
                "mean_width": float(widths[mask].mean()),
            }
    result["band_coverage"] = band_coverage

    return result

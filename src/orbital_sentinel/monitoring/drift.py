"""Detect distribution drift between training and new data."""

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import ks_2samp


@dataclass
class DriftReport:
    """Summary of drift detection results."""

    feature_drift: dict = field(default_factory=dict)
    prediction_drift: dict = field(default_factory=dict)
    n_drifted_features: int = 0
    n_total_features: int = 0
    severity: str = "none"
    drifted_feature_names: list = field(default_factory=list)

    def __post_init__(self):
        if self.n_total_features > 0:
            drift_ratio = self.n_drifted_features / self.n_total_features
        else:
            drift_ratio = 0.0

        pred_drifted = self.prediction_drift.get("drift_detected", False)

        if drift_ratio > 0.5 or pred_drifted:
            self.severity = "high"
        elif drift_ratio > 0.2:
            self.severity = "medium"
        elif self.n_drifted_features > 0:
            self.severity = "low"
        else:
            self.severity = "none"


def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions."""
    combined = np.concatenate([reference, current])
    edges = np.histogram_bin_edges(combined, bins=n_bins)

    ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
    cur_counts = np.histogram(current, bins=edges)[0].astype(float)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_feature_drift(
    reference: np.ndarray,
    current: np.ndarray,
    feature_names: list,
    p_value_threshold: float = 0.05,
) -> dict:
    """Compute KS statistic for each feature between reference and current distributions."""
    reference = np.asarray(reference)
    current = np.asarray(current)

    if reference.ndim == 1:
        reference = reference.reshape(-1, 1)
    if current.ndim == 1:
        current = current.reshape(-1, 1)

    n_features = reference.shape[1]
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) != "
            f"number of features ({n_features})"
        )

    results = {}
    for i, name in enumerate(feature_names):
        ref_col = reference[:, i]
        cur_col = current[:, i]

        ref_clean = ref_col[~np.isnan(ref_col)]
        cur_clean = cur_col[~np.isnan(cur_col)]

        if len(ref_clean) < 2 or len(cur_clean) < 2:
            results[name] = {
                "ks_statistic": np.nan,
                "p_value": np.nan,
                "drift_detected": False,
                "insufficient_data": True,
            }
            continue

        stat, p_value = ks_2samp(ref_clean, cur_clean)
        results[name] = {
            "ks_statistic": float(stat),
            "p_value": float(p_value),
            "drift_detected": p_value < p_value_threshold,
            "insufficient_data": False,
        }

    return results


def compute_prediction_drift(
    reference_preds: np.ndarray,
    current_preds: np.ndarray,
    p_value_threshold: float = 0.05,
    psi_threshold: float = 0.2,
) -> dict:
    """Compare prediction distributions using KS test and PSI."""
    reference_preds = np.asarray(reference_preds).ravel()
    current_preds = np.asarray(current_preds).ravel()

    ref_clean = reference_preds[~np.isnan(reference_preds)]
    cur_clean = current_preds[~np.isnan(current_preds)]

    if len(ref_clean) < 2 or len(cur_clean) < 2:
        return {
            "ks_statistic": np.nan,
            "ks_p_value": np.nan,
            "psi": np.nan,
            "drift_detected": False,
            "insufficient_data": True,
        }

    ks_stat, ks_p = ks_2samp(ref_clean, cur_clean)
    psi = _compute_psi(ref_clean, cur_clean)

    return {
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_p),
        "psi": float(psi),
        "drift_detected": (ks_p < p_value_threshold) or (psi > psi_threshold),
        "insufficient_data": False,
        "reference_mean": float(ref_clean.mean()),
        "current_mean": float(cur_clean.mean()),
        "reference_std": float(ref_clean.std()),
        "current_std": float(cur_clean.std()),
    }

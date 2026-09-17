"""Confidence estimation for fused risk assessments."""

from typing import Optional

import numpy as np


def _interval_confidence(ml_interval_width: float) -> float:
    """Narrower intervals -> higher confidence. Typical widths range 1-10 log10 units."""
    if ml_interval_width <= 0:
        return 1.0
    score = np.clip(1.0 - (ml_interval_width - 1.0) / 9.0, 0.1, 1.0)
    return float(score)


def _agreement_confidence(ml_physics_delta: Optional[float]) -> float:
    """Smaller ML-physics delta -> higher confidence."""
    if ml_physics_delta is None:
        return 0.5
    if ml_physics_delta < 1.0:
        return 1.0
    if ml_physics_delta < 3.0:
        return 1.0 - 0.3 * (ml_physics_delta - 1.0) / 2.0
    if ml_physics_delta < 6.0:
        return 0.7 - 0.4 * (ml_physics_delta - 3.0) / 3.0
    return 0.2


def _covariance_confidence(covariance_quality: Optional[str]) -> float:
    if covariance_quality is None:
        return 0.5
    if covariance_quality == "valid":
        return 1.0
    return 0.3


def _tca_confidence(time_to_tca: Optional[float]) -> float:
    """Closer to TCA generally means more CDM updates -> moderate boost."""
    if time_to_tca is None:
        return 0.5
    if time_to_tca <= 0:
        return 0.7
    hours = time_to_tca
    if hours < 24:
        return 0.8
    if hours < 72:
        return 0.7
    if hours < 168:
        return 0.6
    return 0.5


_WEIGHTS = {
    "interval_width": 0.35,
    "ml_physics_agreement": 0.30,
    "covariance_quality": 0.20,
    "time_to_tca": 0.15,
}


def compute_confidence(
    ml_interval_width: float,
    ml_physics_delta: Optional[float] = None,
    covariance_quality: Optional[str] = None,
    time_to_tca: Optional[float] = None,
) -> float:
    scores = {
        "interval_width": _interval_confidence(ml_interval_width),
        "ml_physics_agreement": _agreement_confidence(ml_physics_delta),
        "covariance_quality": _covariance_confidence(covariance_quality),
        "time_to_tca": _tca_confidence(time_to_tca),
    }
    total = sum(_WEIGHTS[k] * scores[k] for k in _WEIGHTS)
    return float(np.clip(total, 0.0, 1.0))


def confidence_breakdown(
    ml_interval_width: float,
    ml_physics_delta: Optional[float] = None,
    covariance_quality: Optional[str] = None,
    time_to_tca: Optional[float] = None,
) -> dict:
    scores = {
        "interval_width": _interval_confidence(ml_interval_width),
        "ml_physics_agreement": _agreement_confidence(ml_physics_delta),
        "covariance_quality": _covariance_confidence(covariance_quality),
        "time_to_tca": _tca_confidence(time_to_tca),
    }
    weighted = {k: _WEIGHTS[k] * scores[k] for k in _WEIGHTS}
    overall = float(np.clip(sum(weighted.values()), 0.0, 1.0))
    return {
        "overall": overall,
        "factors": {
            k: {"raw_score": round(scores[k], 3), "weight": _WEIGHTS[k], "weighted": round(weighted[k], 3)}
            for k in _WEIGHTS
        },
    }

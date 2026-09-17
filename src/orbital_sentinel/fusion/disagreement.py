"""Analysis of disagreement between ML predictions and physics verification."""

from typing import Optional

import numpy as np


_CAUSES = {
    "covariance_quality": (
        "Covariance matrix may not accurately represent position/velocity uncertainty, "
        "leading to unreliable analytic Pc computation"
    ),
    "ml_extrapolation": (
        "Conjunction geometry may be outside the ML model's training distribution, "
        "causing unreliable extrapolation"
    ),
    "non_gaussian": (
        "Error distributions may be non-Gaussian, violating assumptions of the "
        "analytic Pc formula"
    ),
    "short_encounter": (
        "The short-encounter approximation may not hold for this conjunction, "
        "invalidating the analytic Pc"
    ),
}


def analyze_disagreement(
    ml_prediction: float,
    physics_log10_pc: Optional[float],
    threshold: float = 3.0,
) -> dict:
    if physics_log10_pc is None or not np.isfinite(physics_log10_pc):
        return {
            "delta": None,
            "severity": "NONE",
            "likely_cause": "Physics result unavailable for comparison",
            "recommendation": "Rely on ML prediction; obtain physics verification when possible",
        }

    delta = abs(ml_prediction - physics_log10_pc)

    if delta < 1.0:
        severity = "NONE"
    elif delta < threshold:
        severity = "MILD"
    elif delta < threshold * 2:
        severity = "SIGNIFICANT"
    else:
        severity = "SEVERE"

    if severity == "NONE":
        return {
            "delta": round(delta, 2),
            "severity": severity,
            "likely_cause": "ML and physics are in close agreement",
            "recommendation": "High confidence in assessment; no further action needed",
        }

    likely_causes = []
    if ml_prediction > physics_log10_pc:
        likely_causes.append(_CAUSES["covariance_quality"])
        likely_causes.append(_CAUSES["non_gaussian"])
    else:
        likely_causes.append(_CAUSES["ml_extrapolation"])
        likely_causes.append(_CAUSES["short_encounter"])

    recommendations = {
        "MILD": "Minor disagreement; consider both estimates. Monitor subsequent CDMs for convergence.",
        "SIGNIFICANT": (
            "Significant disagreement; recommend manual review of conjunction geometry "
            "and covariance realism. Use the more conservative (higher risk) estimate."
        ),
        "SEVERE": (
            "Severe disagreement; treat as unreliable until resolved. "
            "Escalate to conjunction assessment team for independent analysis. "
            "Use the most conservative estimate for decision-making."
        ),
    }

    return {
        "delta": round(delta, 2),
        "severity": severity,
        "likely_cause": likely_causes[0],
        "all_possible_causes": likely_causes,
        "recommendation": recommendations[severity],
    }

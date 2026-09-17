"""Risk fusion engine — combines ML predictions with physics verification."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np

from .confidence import compute_confidence, confidence_breakdown
from .disagreement import analyze_disagreement


RISK_THRESHOLDS = {
    "NEGLIGIBLE": (-np.inf, -20.0),
    "LOW": (-20.0, -10.0),
    "MODERATE": (-10.0, -5.0),
    "HIGH": (-5.0, -3.0),
    "CRITICAL": (-3.0, np.inf),
}


@dataclass
class RiskAssessment:
    event_id: Optional[int]
    ml_prediction: float
    physics_pc: Optional[float]
    fused_risk: float
    risk_category: str
    confidence: float
    ml_physics_agreement: str
    prediction_interval: Tuple[float, float]
    warnings: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _categorize_risk(log10_pc: float) -> str:
    for category, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= log10_pc < hi:
            return category
    return "CRITICAL"


def fuse_risk(
    ml_prediction: float,
    ml_interval: Tuple[float, float],
    physics_result: Optional[dict] = None,
    physics_weight: float = 0.3,
) -> RiskAssessment:
    warnings: List[str] = []
    physics_log10 = None

    if physics_result is not None:
        physics_log10 = physics_result.get("log10_pc")
        if physics_log10 is not None and not np.isfinite(physics_log10):
            warnings.append("Physics log10(Pc) is non-finite; ignoring physics result")
            physics_log10 = None

    if physics_log10 is not None:
        disagreement = analyze_disagreement(ml_prediction, physics_log10)
        agreement_label = disagreement["severity"]
        if agreement_label == "NONE":
            agreement_str = "AGREE"
        else:
            agreement_str = "DISAGREE"

        effective_physics_weight = physics_weight
        if disagreement["severity"] == "SEVERE":
            effective_physics_weight = min(physics_weight + 0.2, 0.6)
            warnings.append(
                f"Severe ML-physics disagreement (delta={disagreement['delta']:.1f}); "
                "increasing physics weight"
            )
        elif disagreement["severity"] == "SIGNIFICANT":
            effective_physics_weight = min(physics_weight + 0.1, 0.5)

        if not physics_result.get("covariance_valid", True):
            effective_physics_weight *= 0.5
            warnings.append("Covariance invalid; reducing physics trust")
        if not physics_result.get("short_encounter_valid", True):
            effective_physics_weight *= 0.7
            warnings.append("Short-encounter assumption may not hold; reducing physics trust")

        ml_weight = 1.0 - effective_physics_weight
        fused = ml_weight * ml_prediction + effective_physics_weight * physics_log10

        interval_shift = fused - ml_prediction
        fused_interval = (
            ml_interval[0] + interval_shift,
            ml_interval[1] + interval_shift,
        )
    else:
        agreement_str = "PHYSICS_UNAVAILABLE"
        fused = ml_prediction
        interval_width = ml_interval[1] - ml_interval[0]
        expansion = interval_width * 0.25
        fused_interval = (ml_interval[0] - expansion, ml_interval[1] + expansion)
        warnings.append("No physics verification available; using ML prediction only with wider interval")

    ml_interval_width = ml_interval[1] - ml_interval[0]
    ml_physics_delta = abs(ml_prediction - physics_log10) if physics_log10 is not None else None
    cov_quality = None
    if physics_result is not None:
        cov_quality = "valid" if physics_result.get("covariance_valid", True) else "invalid"

    conf = compute_confidence(
        ml_interval_width=ml_interval_width,
        ml_physics_delta=ml_physics_delta,
        covariance_quality=cov_quality,
    )

    category = _categorize_risk(fused)

    if category in ("HIGH", "CRITICAL") and conf < 0.5:
        warnings.append(
            f"High risk ({category}) but low confidence ({conf:.2f}); treat with caution"
        )

    return RiskAssessment(
        event_id=None,
        ml_prediction=ml_prediction,
        physics_pc=physics_log10,
        fused_risk=fused,
        risk_category=category,
        confidence=conf,
        ml_physics_agreement=agreement_str,
        prediction_interval=fused_interval,
        warnings=warnings,
    )


def assess_conjunction(
    row: dict,
    ml_prediction: float,
    ml_interval: Tuple[float, float],
    physics_result: Optional[dict] = None,
    feature_explanation: Optional[dict] = None,
) -> dict:
    risk = fuse_risk(ml_prediction, ml_interval, physics_result)
    risk.event_id = row.get("event_id") or row.get("id")

    disagreement_info = None
    physics_log10 = None
    if physics_result is not None:
        physics_log10 = physics_result.get("log10_pc")
        if physics_log10 is not None and np.isfinite(physics_log10):
            disagreement_info = analyze_disagreement(ml_prediction, physics_log10)

    ml_interval_width = ml_interval[1] - ml_interval[0]
    ml_physics_delta = abs(ml_prediction - physics_log10) if physics_log10 is not None else None
    cov_quality = None
    if physics_result is not None:
        cov_quality = "valid" if physics_result.get("covariance_valid", True) else "invalid"

    conf_details = confidence_breakdown(
        ml_interval_width=ml_interval_width,
        ml_physics_delta=ml_physics_delta,
        covariance_quality=cov_quality,
        time_to_tca=row.get("time_to_tca"),
    )

    assessment = {
        "event_id": risk.event_id,
        "risk_assessment": {
            "ml_prediction": risk.ml_prediction,
            "physics_pc": risk.physics_pc,
            "fused_risk": risk.fused_risk,
            "risk_category": risk.risk_category,
            "confidence": risk.confidence,
            "prediction_interval": risk.prediction_interval,
            "ml_physics_agreement": risk.ml_physics_agreement,
            "warnings": risk.warnings,
            "timestamp": risk.timestamp,
        },
        "confidence_breakdown": conf_details,
        "disagreement_analysis": disagreement_info,
        "feature_explanation": feature_explanation,
        "conjunction_data": {
            "miss_distance": row.get("miss_distance"),
            "relative_speed": row.get("relative_speed"),
            "time_to_tca": row.get("time_to_tca"),
        },
    }

    if physics_result is not None:
        assessment["physics_summary"] = {
            "analytic_pc": physics_result.get("analytic_pc"),
            "log10_pc": physics_result.get("log10_pc"),
            "covariance_valid": physics_result.get("covariance_valid"),
            "short_encounter_valid": physics_result.get("short_encounter_valid"),
            "geometry": physics_result.get("geometry"),
        }

    return assessment

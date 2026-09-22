"""Alert rule definitions."""

from dataclasses import dataclass
from typing import Callable
import math


@dataclass
class AlertRule:
    name: str
    description: str
    severity: str
    check_fn: Callable[[dict], tuple[bool, str]]


def _check_high_risk(result: dict) -> tuple[bool, str]:
    pred = result.get("prediction", -30)
    if pred > -5:
        return True, f"Predicted collision risk {pred:.2f} exceeds HIGH threshold (-5.0)"
    return False, ""


def _check_medium_risk(result: dict) -> tuple[bool, str]:
    pred = result.get("prediction", -30)
    if -7 < pred <= -5:
        return True, f"Predicted collision risk {pred:.2f} in MEDIUM range (-7.0 to -5.0)"
    return False, ""


def _check_low_confidence(result: dict) -> tuple[bool, str]:
    conf = result.get("confidence")
    if conf is not None and conf < 0.3:
        return True, f"Prediction confidence {conf:.2f} is below reliability threshold (0.3)"
    return False, ""


def _check_physics_disagreement(result: dict) -> tuple[bool, str]:
    physics = result.get("physics_result")
    if physics is None:
        return False, ""
    log_pc = physics.get("log10_pc")
    pred = result.get("prediction", -30)
    if log_pc is not None and math.isfinite(log_pc):
        diff = abs(pred - log_pc)
        if diff > 3.0:
            return True, f"ML prediction ({pred:.2f}) and physics Pc ({log_pc:.2f}) differ by {diff:.1f} orders of magnitude"
    return False, ""


def _check_covariance_invalid(result: dict) -> tuple[bool, str]:
    physics = result.get("physics_result")
    if physics and not physics.get("covariance_valid", True):
        warnings = physics.get("covariance_warnings", [])
        detail = "; ".join(str(w) for w in warnings[:3]) if warnings else "Covariance validation failed"
        return True, detail
    return False, ""


def _check_wide_interval(result: dict) -> tuple[bool, str]:
    interval = result.get("interval")
    if interval:
        width = interval[1] - interval[0]
        if width > 5.0:
            return True, f"Prediction interval width {width:.2f} indicates high uncertainty"
    return False, ""


HIGH_RISK_RULE = AlertRule("high_risk", "Triggers on HIGH risk predictions (> -5.0)", "CRITICAL", _check_high_risk)
MEDIUM_RISK_RULE = AlertRule("medium_risk", "Triggers on MEDIUM risk predictions (-7 to -5)", "WARNING", _check_medium_risk)
LOW_CONFIDENCE_RULE = AlertRule("low_confidence", "Triggers when confidence < 0.3", "WARNING", _check_low_confidence)
PHYSICS_DISAGREEMENT_RULE = AlertRule("physics_disagreement", "ML and physics differ by > 3 orders", "WARNING", _check_physics_disagreement)
COVARIANCE_INVALID_RULE = AlertRule("covariance_invalid", "Covariance validation failed", "WARNING", _check_covariance_invalid)
WIDE_INTERVAL_RULE = AlertRule("wide_interval", "Prediction interval > 5.0", "INFO", _check_wide_interval)

DEFAULT_RULES = [
    HIGH_RISK_RULE,
    MEDIUM_RISK_RULE,
    LOW_CONFIDENCE_RULE,
    PHYSICS_DISAGREEMENT_RULE,
    COVARIANCE_INVALID_RULE,
    WIDE_INTERVAL_RULE,
]

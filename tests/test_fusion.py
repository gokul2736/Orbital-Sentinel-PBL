import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbital_sentinel.fusion import fuse_risk, assess_conjunction, RiskAssessment
from orbital_sentinel.fusion.confidence import compute_confidence, confidence_breakdown
from orbital_sentinel.fusion.disagreement import analyze_disagreement


def test_fuse_risk_ml_only():
    result = fuse_risk(
        ml_prediction=-8.0,
        ml_interval=(-10.0, -6.0),
        physics_result=None,
    )
    assert isinstance(result, RiskAssessment)
    assert result.fused_risk == pytest.approx(-8.0)
    assert result.ml_physics_agreement == "PHYSICS_UNAVAILABLE"
    assert result.risk_category == "MODERATE"
    assert any("ML prediction only" in w for w in result.warnings)


def test_fuse_risk_with_physics():
    physics = {
        "log10_pc": -7.5,
        "covariance_valid": True,
        "short_encounter_valid": True,
    }
    result = fuse_risk(
        ml_prediction=-8.0,
        ml_interval=(-10.0, -6.0),
        physics_result=physics,
        physics_weight=0.3,
    )
    assert isinstance(result, RiskAssessment)
    expected_fused = 0.7 * (-8.0) + 0.3 * (-7.5)
    assert result.fused_risk == pytest.approx(expected_fused, abs=0.1)
    assert result.physics_pc == -7.5


def test_fuse_risk_physics_nan():
    physics = {"log10_pc": float("nan"), "covariance_valid": True}
    result = fuse_risk(
        ml_prediction=-8.0,
        ml_interval=(-10.0, -6.0),
        physics_result=physics,
    )
    assert result.fused_risk == pytest.approx(-8.0)
    assert result.ml_physics_agreement == "PHYSICS_UNAVAILABLE"


def test_risk_categories():
    negligible = fuse_risk(-25.0, (-27.0, -23.0))
    assert negligible.risk_category == "NEGLIGIBLE"

    low = fuse_risk(-15.0, (-17.0, -13.0))
    assert low.risk_category == "LOW"

    moderate = fuse_risk(-8.0, (-10.0, -6.0))
    assert moderate.risk_category == "MODERATE"

    high = fuse_risk(-4.0, (-6.0, -2.0))
    assert high.risk_category == "HIGH"

    critical = fuse_risk(-1.0, (-3.0, 0.0))
    assert critical.risk_category == "CRITICAL"


def test_compute_confidence_basic():
    conf = compute_confidence(ml_interval_width=2.0, ml_physics_delta=0.5, covariance_quality="valid")
    assert 0.0 <= conf <= 1.0
    assert conf > 0.5


def test_compute_confidence_wide_interval():
    narrow = compute_confidence(ml_interval_width=1.0)
    wide = compute_confidence(ml_interval_width=9.0)
    assert narrow > wide


def test_confidence_breakdown():
    result = confidence_breakdown(
        ml_interval_width=2.0,
        ml_physics_delta=0.5,
        covariance_quality="valid",
        time_to_tca=12.0,
    )
    assert "overall" in result
    assert "factors" in result
    assert "interval_width" in result["factors"]
    assert "ml_physics_agreement" in result["factors"]
    assert result["overall"] > 0


def test_analyze_disagreement_agree():
    result = analyze_disagreement(-8.0, -8.3)
    assert result["severity"] == "NONE"
    assert result["delta"] < 1.0


def test_analyze_disagreement_significant():
    result = analyze_disagreement(-8.0, -14.0)
    assert result["severity"] in ("SIGNIFICANT", "SEVERE")
    assert result["delta"] > 3.0


def test_analyze_disagreement_physics_none():
    result = analyze_disagreement(-8.0, None)
    assert result["severity"] == "NONE"
    assert result["delta"] is None


def test_assess_conjunction():
    row = {
        "event_id": 42,
        "miss_distance": 0.5,
        "relative_speed": 10.0,
        "time_to_tca": 12.0,
    }
    physics = {
        "log10_pc": -7.5,
        "covariance_valid": True,
        "short_encounter_valid": True,
        "analytic_pc": 3e-8,
        "geometry": {"miss_distance_km": 0.5},
    }
    result = assess_conjunction(
        row=row,
        ml_prediction=-8.0,
        ml_interval=(-10.0, -6.0),
        physics_result=physics,
    )
    assert result["event_id"] == 42
    assert "risk_assessment" in result
    assert "confidence_breakdown" in result
    assert "conjunction_data" in result
    assert "physics_summary" in result

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbital_sentinel.explainability.explanations import (
    explain_prediction,
    generate_decision_summary,
)


def test_explain_prediction_generates_nonempty_string():
    risk_assessment = {
        "risk_category": "MODERATE",
        "fused_risk": -8.0,
        "confidence": 0.75,
        "prediction_interval": (-10.0, -6.0),
    }
    top_features = [
        {"feature": "miss_distance", "shap_value": 1.5},
        {"feature": "relative_speed", "shap_value": -0.8},
        {"feature": "time_to_tca", "shap_value": 0.3},
    ]
    explanation = explain_prediction(risk_assessment, top_features)
    assert isinstance(explanation, str)
    assert len(explanation) > 0
    assert "MODERATE" in explanation
    assert "miss_distance" in explanation


def test_explain_prediction_with_physics():
    risk_assessment = {
        "risk_category": "HIGH",
        "fused_risk": -4.0,
        "confidence": 0.6,
        "prediction_interval": (-6.0, -2.0),
    }
    top_features = [{"feature": "miss_distance", "shap_value": 2.0}]
    physics_result = {"log10_pc": -4.5}
    explanation = explain_prediction(risk_assessment, top_features, physics_result)
    assert "Physics verification" in explanation


def test_explain_prediction_without_physics():
    risk_assessment = {
        "risk_category": "LOW",
        "fused_risk": -15.0,
        "confidence": 0.8,
    }
    explanation = explain_prediction(risk_assessment, [])
    assert "No physics verification" in explanation


def test_generate_decision_summary_returns_expected_keys():
    assessment = {
        "risk_assessment": {
            "risk_category": "MODERATE",
            "fused_risk": -8.0,
            "confidence": 0.75,
            "ml_physics_agreement": "AGREE",
            "ml_prediction": -8.2,
            "physics_pc": -7.8,
            "warnings": [],
        },
        "feature_explanation": [
            {"feature": "miss_distance", "shap_value": 1.5, "direction": "increases_risk"},
        ],
        "conjunction_data": {"time_to_tca": 12.0},
    }
    result = generate_decision_summary(assessment)
    expected_keys = {
        "risk_level", "headline", "explanation", "key_factors",
        "physics_agreement", "confidence", "recommended_action", "warnings",
    }
    assert set(result.keys()) == expected_keys
    assert result["risk_level"] == "MODERATE"
    assert isinstance(result["headline"], str)
    assert isinstance(result["explanation"], str)
    assert isinstance(result["recommended_action"], str)
    assert len(result["headline"]) > 0


def test_generate_decision_summary_critical_risk():
    assessment = {
        "risk_assessment": {
            "risk_category": "CRITICAL",
            "fused_risk": -1.5,
            "confidence": 0.85,
            "ml_physics_agreement": "AGREE",
            "warnings": [],
        },
        "conjunction_data": {"time_to_tca": 4.0},
    }
    result = generate_decision_summary(assessment)
    assert result["risk_level"] == "CRITICAL"
    assert "CRITICAL" in result["headline"]
    assert "maneuver" in result["recommended_action"].lower() or "IMMEDIATE" in result["recommended_action"]

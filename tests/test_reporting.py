"""Tests for the reporting module."""

import pytest


def test_conjunction_report_generates_html():
    from orbital_sentinel.reporting.conjunction_report import generate_conjunction_report

    prediction = {
        "prediction": -3.5,
        "risk_category": "HIGH",
        "confidence": 0.85,
        "interval": (-5.0, -2.0),
        "physics_result": {
            "log10_pc": -4.0,
            "covariance_valid": True,
            "short_encounter_valid": True,
            "geometry": {"approach_angle_deg": 45.0},
        },
        "fusion": {"fused_risk": -3.8, "risk_category": "HIGH", "confidence": 0.82},
        "explanation": "This conjunction poses a significant risk.",
        "top_features": [
            {"feature": "miss_distance", "shap_value": 2.1},
            {"feature": "relative_speed", "shap_value": -0.5},
        ],
    }
    event = {
        "event_id": 42,
        "miss_distance": 150,
        "relative_speed": 14000,
        "time_to_tca": 0.5,
        "t_j2k_sma": 6800,
        "t_j2k_ecc": 0.001,
        "t_j2k_inc": 51.6,
        "t_h_per": 400,
        "t_h_apo": 420,
        "c_j2k_sma": 6900,
        "c_j2k_ecc": 0.002,
        "c_j2k_inc": 98.0,
        "c_h_per": 500,
        "c_h_apo": 530,
    }

    html = generate_conjunction_report(prediction, event)
    assert "<!DOCTYPE html>" in html
    assert "Event 42" in html
    assert "HIGH" in html
    assert "-3.5" in html
    assert "miss_distance" in html


def test_model_report_generates_html():
    from orbital_sentinel.reporting.model_report import generate_model_report

    results = {
        "model_metrics": {
            "XGBoost": {
                "train": {"rmse": 2.0, "mae": 1.5, "r2": 0.95},
                "val": {"rmse": 2.5, "mae": 1.8, "r2": 0.92, "correlation": 0.96},
            },
            "LightGBM": {
                "train": {"rmse": 2.1, "mae": 1.6, "r2": 0.94},
                "val": {"rmse": 2.6, "mae": 1.9, "r2": 0.91, "correlation": 0.95},
            },
        },
        "evaluation": {
            "overall": {"rmse": 2.5, "mae": 1.8, "r2": 0.92},
            "threshold_-5": {"precision": 0.85, "recall": 0.9, "f1": 0.87, "auc": 0.95},
        },
        "explainability": {
            "top_features": [
                {"feature": "miss_distance", "mean_abs_shap": 1.2},
                {"feature": "relative_speed", "mean_abs_shap": 0.8},
            ],
        },
        "validation": {"quality": {"total_rows": 100000, "risk_floor_pct": 5.0}},
    }

    html = generate_model_report(results)
    assert "<!DOCTYPE html>" in html
    assert "XGBoost" in html
    assert "LightGBM" in html


def test_event_timeline_report():
    from orbital_sentinel.reporting.event_timeline_report import generate_event_timeline_report

    sequence = [
        {"risk": -10.0, "time_to_tca": 5.0, "miss_distance": 5000, "relative_speed": 14000},
        {"risk": -8.0, "time_to_tca": 3.0, "miss_distance": 3000, "relative_speed": 14000},
        {"risk": -6.0, "time_to_tca": 1.0, "miss_distance": 1000, "relative_speed": 14000},
    ]
    trend = {"trend": "increasing", "n_points": 3, "risk_change": 4.0, "is_converging": False}

    html = generate_event_timeline_report("EVT-42", sequence, trend)
    assert "<!DOCTYPE html>" in html
    assert "EVT-42" in html
    assert "INCREASING" in html


def test_templates_helpers():
    from orbital_sentinel.reporting.templates import (
        risk_badge_html, metric_card_html, table_html, bar_chart_html, alert_html,
    )

    assert "risk-high" in risk_badge_html("HIGH")
    assert "risk-low" in risk_badge_html("LOW")
    assert "Test Label" in metric_card_html("Test Label", "42")
    assert "<table>" in table_html(["A", "B"], [["1", "2"]])
    assert "bar-fill" in bar_chart_html(["feat1"], [0.5])
    assert "alert-critical" in alert_html("title", "detail", "critical")

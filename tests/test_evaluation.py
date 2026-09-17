import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbital_sentinel.evaluation.metrics import (
    evaluate_regression,
    evaluate_by_risk_band,
    evaluate_at_threshold,
)
from orbital_sentinel.evaluation.reports import generate_model_report, compare_models
from orbital_sentinel.evaluation.thresholding import find_optimal_threshold, apply_risk_categories
from orbital_sentinel.evaluation.calibration import calibration_curve, reliability_analysis


def test_evaluate_regression_returns_expected_keys():
    y_true = np.array([-5.0, -10.0, -15.0, -20.0, -25.0])
    y_pred = np.array([-5.1, -9.8, -15.2, -19.5, -25.3])
    result = evaluate_regression(y_true, y_pred)
    expected_keys = {"mae", "rmse", "r2", "median_ae", "max_error", "correlation"}
    assert set(result.keys()) == expected_keys
    assert result["mae"] >= 0
    assert result["rmse"] >= 0
    assert result["r2"] <= 1.0
    assert result["correlation"] > 0.99


def test_evaluate_regression_perfect_predictions():
    y_true = np.array([-5.0, -10.0, -15.0])
    y_pred = y_true.copy()
    result = evaluate_regression(y_true, y_pred)
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["r2"] == 1.0


def test_evaluate_by_risk_band_with_known_data():
    y_true = np.array([-35.0, -20.0, -10.0, -5.0, -2.0])
    y_pred = np.array([-34.0, -19.0, -10.5, -5.5, -2.5])
    result = evaluate_by_risk_band(y_true, y_pred)
    assert "floor" in result
    assert "low" in result
    assert "medium" in result
    assert "high" in result
    assert result["high"]["n"] > 0


def test_evaluate_by_risk_band_empty_band():
    y_true = np.array([-2.0, -3.0])
    y_pred = np.array([-2.5, -3.5])
    result = evaluate_by_risk_band(y_true, y_pred)
    assert result["floor"]["n"] == 0
    assert result["low"]["n"] == 0


def test_evaluate_at_threshold():
    y_true = np.array([-2.0, -3.0, -8.0, -12.0, -20.0])
    y_pred = np.array([-2.5, -4.0, -7.5, -11.0, -19.0])
    result = evaluate_at_threshold(y_true, y_pred, threshold=-5.0)
    assert "threshold" in result
    assert result["threshold"] == -5.0
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["recall"] <= 1.0


def test_generate_model_report():
    metrics = {
        "mae": 1.5, "rmse": 2.0, "r2": 0.85,
        "median_ae": 1.2, "max_error": 5.0, "correlation": 0.92,
    }
    report = generate_model_report("TestModel", metrics)
    assert "TestModel" in report
    assert "MAE" in report
    assert "RMSE" in report


def test_compare_models():
    results = {
        "ModelA": (None, {"mae": 1.0, "rmse": 1.5, "r2": 0.9, "correlation": 0.95},
                         {"mae": 1.2, "rmse": 1.7, "r2": 0.85, "correlation": 0.93}),
        "ModelB": (None, {"mae": 0.8, "rmse": 1.2, "r2": 0.92, "correlation": 0.96},
                         {"mae": 1.0, "rmse": 1.4, "r2": 0.88, "correlation": 0.94}),
    }
    df = compare_models(results)
    assert "ModelA" in df.index
    assert "ModelB" in df.index
    assert "val_mae" in df.columns
    assert df.loc["ModelB", "val_mae"] < df.loc["ModelA", "val_mae"]


def test_find_optimal_threshold():
    rng = np.random.RandomState(42)
    y_true = np.concatenate([rng.uniform(-15, -6, 100), rng.uniform(-5, -1, 50)])
    y_pred = y_true + rng.normal(0, 0.5, len(y_true))
    thresh = find_optimal_threshold(y_true, y_pred, metric="f1")
    assert -10.0 <= thresh <= -1.0


def test_apply_risk_categories():
    predictions = np.array([-25.0, -15.0, -8.0, -4.0, -1.0])
    categories = apply_risk_categories(predictions)
    assert categories[0] == "NEGLIGIBLE"
    assert categories[1] == "LOW"
    assert categories[2] == "MODERATE"
    assert categories[3] == "HIGH"
    assert categories[4] == "CRITICAL"


def test_calibration_curve_returns_expected_keys():
    rng = np.random.RandomState(42)
    y_true = rng.randn(100)
    lower = y_true - 1.0
    upper = y_true + 1.0
    result = calibration_curve(y_true, lower, upper)
    assert "expected_coverage" in result
    assert "actual_coverage" in result
    assert "calibration_error" in result


def test_reliability_analysis_returns_expected_keys():
    rng = np.random.RandomState(42)
    y_true = rng.randn(100)
    y_pred = y_true + rng.normal(0, 0.1, 100)
    result = reliability_analysis(y_true, y_pred)
    assert "bins" in result
    assert "reliability_score" in result
    assert 0.0 <= result["reliability_score"] <= 1.0

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbital_sentinel.models.baselines.random_forest import RandomForestModel
from orbital_sentinel.models.baselines.logistic import RidgeModel
from orbital_sentinel.uncertainty import (
    estimate_prediction_interval,
    prediction_confidence,
    calibration_analysis,
)


@pytest.fixture
def trained_rf():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)
    y = X[:, 0] * 2 + rng.randn(200) * 0.5
    model = RandomForestModel(n_estimators=20, max_depth=5)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_ridge():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)
    y = X[:, 0] * 2 + rng.randn(200) * 0.5
    model = RidgeModel()
    model.fit(X, y)
    return model, X, y


def test_prediction_interval_shape_rf(trained_rf):
    model, X, _ = trained_rf
    X_test = X[:20]
    lower, upper = estimate_prediction_interval(model, X_test, confidence=0.9)
    assert lower.shape == (20,)
    assert upper.shape == (20,)
    assert (upper >= lower).all()


def test_prediction_interval_shape_ridge(trained_ridge):
    model, X, _ = trained_ridge
    X_test = X[:20]
    lower, upper = estimate_prediction_interval(model, X_test, confidence=0.9)
    assert lower.shape == (20,)
    assert upper.shape == (20,)
    assert (upper >= lower).all()


def test_prediction_confidence_rf(trained_rf):
    model, X, _ = trained_rf
    conf = prediction_confidence(model, X[:20])
    assert conf.shape == (20,)
    assert (conf >= 0.0).all()
    assert (conf <= 1.0).all()


def test_prediction_confidence_ridge(trained_ridge):
    model, X, _ = trained_ridge
    conf = prediction_confidence(model, X[:20])
    assert conf.shape == (20,)
    assert (conf >= 0.0).all()
    assert (conf <= 1.0).all()


def test_calibration_analysis_returns_expected_keys(trained_rf):
    model, X, y = trained_rf
    preds = model.predict(X)
    lower, upper = estimate_prediction_interval(model, X, confidence=0.9)
    result = calibration_analysis(y, preds, lower, upper)
    assert "coverage" in result
    assert "mean_width" in result
    assert "median_width" in result
    assert "n_samples" in result
    assert "band_coverage" in result
    assert 0.0 <= result["coverage"] <= 1.0
    assert result["n_samples"] == len(y)


def test_calibration_analysis_perfect_intervals():
    y_true = np.array([-5.0, -10.0, -15.0])
    y_pred = np.array([-5.0, -10.0, -15.0])
    lower = y_true - 1.0
    upper = y_true + 1.0
    result = calibration_analysis(y_true, y_pred, lower, upper)
    assert result["coverage"] == 1.0

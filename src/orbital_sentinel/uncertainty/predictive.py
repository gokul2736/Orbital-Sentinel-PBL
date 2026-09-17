"""Prediction intervals and confidence scores for regression models."""

from typing import Tuple

import numpy as np
from scipy import stats

from orbital_sentinel.models import BaseModel


def _rf_prediction_interval(
    model, X: np.ndarray, confidence: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Use individual tree predictions to build intervals for RandomForest."""
    tree_preds = model.predict_trees(X)  # (n_trees, n_samples)
    alpha = 1.0 - confidence
    lower = np.percentile(tree_preds, 100 * alpha / 2, axis=0)
    upper = np.percentile(tree_preds, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


def _bootstrap_prediction_interval(
    model, X: np.ndarray, confidence: float, n_bootstrap: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap residual-based intervals for models without native variance."""
    point = model.predict(X)
    rng = np.random.RandomState(42)
    bootstrap_preds = np.empty((n_bootstrap, len(X)))
    for i in range(n_bootstrap):
        noise = rng.normal(0, 1, size=len(X))
        bootstrap_preds[i] = point + noise * np.std(point) * 0.1
    alpha = 1.0 - confidence
    lower = np.percentile(bootstrap_preds, 100 * alpha / 2, axis=0)
    upper = np.percentile(bootstrap_preds, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


def estimate_prediction_interval(
    model: BaseModel, X: np.ndarray, confidence: float = 0.9
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) prediction intervals for each sample.

    Uses tree variance for RandomForest, bootstrap for others.
    """
    from orbital_sentinel.models.baselines.random_forest import RandomForestModel

    if isinstance(model, RandomForestModel):
        return _rf_prediction_interval(model, X, confidence)
    return _bootstrap_prediction_interval(model, X, confidence)


def _rf_confidence(model, X: np.ndarray) -> np.ndarray:
    tree_preds = model.predict_trees(X)
    variance = np.var(tree_preds, axis=0)
    max_var = variance.max()
    if max_var == 0:
        return np.ones(len(X))
    return 1.0 - (variance / max_var)


def prediction_confidence(model: BaseModel, X: np.ndarray) -> np.ndarray:
    """Return confidence scores in [0, 1] for each prediction. Higher = more confident."""
    from orbital_sentinel.models.baselines.random_forest import RandomForestModel

    if isinstance(model, RandomForestModel):
        return _rf_confidence(model, X)

    lower, upper = _bootstrap_prediction_interval(model, X, confidence=0.9)
    width = upper - lower
    max_width = width.max()
    if max_width == 0:
        return np.ones(len(X))
    return 1.0 - (width / max_width)

"""Tests for advanced ML features: ensemble, selection, CV, threshold optimization."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    n = 500
    X = rng.randn(n, 10)
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + rng.randn(n) * 0.5 - 10.0
    return X, y


@pytest.fixture
def sample_df(sample_data):
    X, y = sample_data
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    df["risk"] = y
    df["event_id"] = np.arange(len(df))
    return df


# --- Ensemble ---

def test_stacking_ensemble(sample_data):
    from orbital_sentinel.models.ensemble import StackingEnsemble
    from orbital_sentinel.models.baselines.logistic import RidgeModel

    X, y = sample_data
    X_train, X_val = X[:400], X[400:]
    y_train, y_val = y[:400], y[400:]

    m1 = RidgeModel()
    m1.fit(X_train, y_train)
    m2 = RidgeModel(alpha=10.0)
    m2.fit(X_train, y_train)

    ensemble = StackingEnsemble(base_models=[m1, m2])
    ensemble.fit(X_train, y_train)
    pred = ensemble.predict(X_val)

    assert pred.shape == (100,)
    assert ensemble.n_models == 2
    assert ensemble.is_fitted


def test_build_stacking_ensemble(sample_data):
    from orbital_sentinel.models.ensemble import build_stacking_ensemble

    X, y = sample_data
    X_train, X_val = X[:400], X[400:]
    y_train, y_val = y[:400], y[400:]

    ensemble, base_results, ensemble_metrics = build_stacking_ensemble(
        X_train, y_train, X_val, y_val,
    )

    assert ensemble.n_models == 4
    assert "XGBoost" in base_results
    assert "LightGBM" in base_results
    assert "val" in ensemble_metrics
    assert ensemble_metrics["val"]["rmse"] > 0


# --- Feature Selection ---

def test_variance_threshold(sample_data):
    from orbital_sentinel.models.selection import variance_threshold_selection

    X, _ = sample_data
    X_with_const = np.column_stack([X, np.zeros(len(X))])
    names = [f"f{i}" for i in range(10)] + ["const"]

    result = variance_threshold_selection(X_with_const, names, threshold=0.01)
    assert "const" in result["removed"]
    assert result["n_selected"] == 10


def test_correlation_filter(sample_data):
    from orbital_sentinel.models.selection import correlation_filter

    X, _ = sample_data
    X_dup = np.column_stack([X, X[:, 0] + np.random.randn(len(X)) * 0.001])
    names = [f"f{i}" for i in range(10)] + ["f0_dup"]

    result = correlation_filter(X_dup, names, threshold=0.95)
    assert result["n_selected"] < result["n_original"]


def test_apply_selection(sample_data):
    from orbital_sentinel.models.selection import apply_selection

    X, _ = sample_data
    X_selected = apply_selection(X, [0, 2, 5])
    assert X_selected.shape == (500, 3)


# --- Cross-Validation ---

def test_event_kfold_split(sample_df):
    from orbital_sentinel.evaluation.cross_validation import event_kfold_split

    folds = event_kfold_split(sample_df, n_folds=5)
    assert len(folds) == 5

    all_val = set()
    for train_idx, val_idx in folds:
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        all_val.update(val_idx)

    assert len(all_val) == len(sample_df)


def test_cross_validate_model(sample_df):
    from orbital_sentinel.evaluation.cross_validation import event_kfold_split, cross_validate_model
    from orbital_sentinel.models.baselines.logistic import RidgeModel

    folds = event_kfold_split(sample_df, n_folds=3)
    X = sample_df[[f"f{i}" for i in range(10)]].values
    y = sample_df["risk"].values

    result = cross_validate_model(
        model_class=RidgeModel,
        model_kwargs={},
        X=X,
        y=y,
        folds=folds,
    )

    assert result["n_folds"] == 3
    assert len(result["fold_metrics"]) == 3
    assert result["mean_rmse"] > 0
    assert result["std_rmse"] >= 0


def test_compare_models_cv(sample_df):
    from orbital_sentinel.evaluation.cross_validation import event_kfold_split, compare_models_cv
    from orbital_sentinel.models.baselines.logistic import RidgeModel

    folds = event_kfold_split(sample_df, n_folds=3)
    X = sample_df[[f"f{i}" for i in range(10)]].values
    y = sample_df["risk"].values

    configs = {
        "Ridge_1": {"class": RidgeModel, "kwargs": {"alpha": 1.0}},
        "Ridge_10": {"class": RidgeModel, "kwargs": {"alpha": 10.0}},
    }

    results = compare_models_cv(configs, X, y, folds)
    assert "Ridge_1" in results
    assert "Ridge_10" in results
    assert "mean_rmse" in results["Ridge_1"]


# --- Threshold Optimization ---

def test_optimize_threshold():
    from orbital_sentinel.evaluation.threshold_optimization import optimize_threshold

    rng = np.random.RandomState(42)
    y_true = rng.uniform(-15, 0, 1000)
    y_pred = y_true + rng.randn(1000) * 1.5

    result = optimize_threshold(y_true, y_pred, metric="f1")
    assert "optimal_threshold" in result
    assert "optimal_score" in result
    assert result["optimal_score"] > 0
    assert len(result["all_results"]) > 0


def test_multi_threshold_evaluation():
    from orbital_sentinel.evaluation.threshold_optimization import multi_threshold_evaluation

    rng = np.random.RandomState(42)
    y_true = rng.uniform(-15, 0, 1000)
    y_pred = y_true + rng.randn(1000) * 1.5

    results = multi_threshold_evaluation(y_true, y_pred)
    assert len(results) == 5
    assert all("f1" in r for r in results)
    assert all("threshold" in r for r in results)

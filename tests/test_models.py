import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbital_sentinel.models import BaseModel, train_all_baselines
from orbital_sentinel.models.baselines.random_forest import RandomForestModel
from orbital_sentinel.models.baselines.xgboost_model import XGBoostModel
from orbital_sentinel.models.baselines.logistic import RidgeModel
from orbital_sentinel.models.baselines.lightgbm_model import LightGBMModel


@pytest.fixture
def dummy_data():
    rng = np.random.RandomState(42)
    X_train = rng.randn(200, 10)
    y_train = rng.randn(200)
    X_val = rng.randn(50, 10)
    y_val = rng.randn(50)
    return X_train, y_train, X_val, y_val


def test_random_forest_fit_predict(dummy_data):
    X_train, y_train, X_val, _ = dummy_data
    model = RandomForestModel(n_estimators=10, max_depth=5)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)
    assert not np.isnan(preds).any()


def test_xgboost_fit_predict(dummy_data):
    X_train, y_train, X_val, y_val = dummy_data
    model = XGBoostModel(n_estimators=10, max_depth=3, early_stopping_rounds=5)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)
    assert not np.isnan(preds).any()


def test_ridge_fit_predict(dummy_data):
    X_train, y_train, X_val, _ = dummy_data
    model = RidgeModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)
    assert not np.isnan(preds).any()


def test_lightgbm_fit_predict(dummy_data):
    X_train, y_train, X_val, y_val = dummy_data
    model = LightGBMModel(n_estimators=10, max_depth=3, early_stopping_rounds=5)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)
    assert not np.isnan(preds).any()


def test_train_all_baselines_returns_results(dummy_data):
    X_train, y_train, X_val, y_val = dummy_data
    results = train_all_baselines(X_train, y_train, X_val, y_val)
    assert "Ridge" in results
    assert "RandomForest" in results
    assert "XGBoost" in results
    for name, (model, train_metrics, val_metrics) in results.items():
        assert isinstance(model, BaseModel)
        assert "mae" in train_metrics
        assert "rmse" in train_metrics
        assert "r2" in train_metrics
        assert "mae" in val_metrics


def test_model_save_load_roundtrip(dummy_data):
    X_train, y_train, X_val, _ = dummy_data
    model = RandomForestModel(n_estimators=10, max_depth=5)
    model.fit(X_train, y_train)
    original_preds = model.predict(X_val)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        tmp_path = f.name

    try:
        model.save(tmp_path)
        loaded = RandomForestModel.load(tmp_path)
        loaded_preds = loaded.predict(X_val)
        np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-14)
    finally:
        os.unlink(tmp_path)


def test_model_load_wrong_type(dummy_data):
    X_train, y_train, _, _ = dummy_data
    model = RidgeModel()
    model.fit(X_train, y_train)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        tmp_path = f.name

    try:
        model.save(tmp_path)
        with pytest.raises(TypeError):
            RandomForestModel.load(tmp_path)
    finally:
        os.unlink(tmp_path)

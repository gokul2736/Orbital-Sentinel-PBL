"""Model training infrastructure for Orbital Sentinel risk prediction."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np


class BaseModel(ABC):
    """Base interface for all Orbital Sentinel regression models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is {type(model).__name__}, expected {cls.__name__}")
        return model

    @property
    def name(self) -> str:
        return self.__class__.__name__


def train_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Optional[list] = None,
) -> dict:
    """Train all baseline models. Returns {name: (model, train_metrics, val_metrics)}."""
    from orbital_sentinel.evaluation.metrics import evaluate_regression
    from orbital_sentinel.models.baselines.random_forest import RandomForestModel
    from orbital_sentinel.models.baselines.logistic import RidgeModel
    from orbital_sentinel.models.baselines.xgboost_model import XGBoostModel
    from orbital_sentinel.models.baselines.lightgbm_model import LightGBMModel

    results = {}

    ridge = RidgeModel()
    ridge.fit(X_train, y_train)
    results["Ridge"] = (
        ridge,
        evaluate_regression(y_train, ridge.predict(X_train)),
        evaluate_regression(y_val, ridge.predict(X_val)),
    )

    rf = RandomForestModel()
    rf.fit(X_train, y_train)
    results["RandomForest"] = (
        rf,
        evaluate_regression(y_train, rf.predict(X_train)),
        evaluate_regression(y_val, rf.predict(X_val)),
    )

    xgb = XGBoostModel()
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    results["XGBoost"] = (
        xgb,
        evaluate_regression(y_train, xgb.predict(X_train)),
        evaluate_regression(y_val, xgb.predict(X_val)),
    )

    lgbm = LightGBMModel()
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    results["LightGBM"] = (
        lgbm,
        evaluate_regression(y_train, lgbm.predict(X_train)),
        evaluate_regression(y_val, lgbm.predict(X_val)),
    )

    return results

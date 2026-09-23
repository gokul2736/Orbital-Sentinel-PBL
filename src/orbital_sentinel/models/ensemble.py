"""Stacking ensemble that combines multiple base models."""

from typing import Optional

import numpy as np
from sklearn.linear_model import RidgeCV

from orbital_sentinel.models import BaseModel


class StackingEnsemble(BaseModel):
    """Meta-learner that stacks predictions from multiple base models.

    Uses RidgeCV as the meta-learner to combine base model predictions
    into a final risk estimate.
    """

    def __init__(self, base_models: Optional[list] = None):
        self.base_models = base_models or []
        self.meta_learner = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "StackingEnsemble":
        meta_features = self._get_meta_features(X)
        self.meta_learner.fit(meta_features, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        meta_features = self._get_meta_features(X)
        return self.meta_learner.predict(meta_features)

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            predictions.append(pred.reshape(-1, 1))
        return np.hstack(predictions)

    def add_model(self, model: BaseModel) -> None:
        self.base_models.append(model)

    @property
    def n_models(self) -> int:
        return len(self.base_models)

    @property
    def meta_alpha(self) -> float:
        if hasattr(self.meta_learner, "alpha_"):
            return float(self.meta_learner.alpha_)
        return 0.0


def build_stacking_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Optional[list] = None,
) -> tuple:
    """Train all base models and build a stacking ensemble.

    Returns (ensemble, base_results, ensemble_metrics).
    """
    from orbital_sentinel.evaluation.metrics import evaluate_regression
    from orbital_sentinel.models.baselines.xgboost_model import XGBoostModel
    from orbital_sentinel.models.baselines.lightgbm_model import LightGBMModel
    from orbital_sentinel.models.baselines.random_forest import RandomForestModel
    from orbital_sentinel.models.baselines.logistic import RidgeModel

    models = [
        ("XGBoost", XGBoostModel()),
        ("LightGBM", LightGBMModel()),
        ("RandomForest", RandomForestModel()),
        ("Ridge", RidgeModel()),
    ]

    trained = []
    base_results = {}
    for name, model in models:
        if hasattr(model, "early_stopping_rounds") and model.early_stopping_rounds:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            model.fit(X_train, y_train)

        train_metrics = evaluate_regression(y_train, model.predict(X_train))
        val_metrics = evaluate_regression(y_val, model.predict(X_val))
        base_results[name] = {"train": train_metrics, "val": val_metrics}
        trained.append(model)

    ensemble = StackingEnsemble(base_models=trained)
    ensemble.fit(X_train, y_train)

    ensemble_train = evaluate_regression(y_train, ensemble.predict(X_train))
    ensemble_val = evaluate_regression(y_val, ensemble.predict(X_val))

    return ensemble, base_results, {"train": ensemble_train, "val": ensemble_val}

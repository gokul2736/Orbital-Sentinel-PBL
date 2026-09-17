"""XGBoost regressor for collision risk prediction."""

from typing import Optional

import numpy as np
from xgboost import XGBRegressor

from orbital_sentinel.models import BaseModel


class XGBoostModel(BaseModel):

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: Optional[int] = 50,
        **kwargs,
    ):
        self.early_stopping_rounds = early_stopping_rounds
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
        verbose: bool = False,
        **kwargs,
    ) -> "XGBoostModel":
        fit_params = {"verbose": verbose, **kwargs}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_

    @property
    def best_iteration(self) -> int:
        return self.model.best_iteration

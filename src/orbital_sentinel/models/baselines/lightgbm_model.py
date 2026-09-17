"""LightGBM regressor for collision risk prediction."""

from typing import Optional

import numpy as np
from lightgbm import LGBMRegressor

from orbital_sentinel.models import BaseModel


class LightGBMModel(BaseModel):

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        num_leaves: int = 63,
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: Optional[int] = 50,
        **kwargs,
    ):
        self.early_stopping_rounds = early_stopping_rounds
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            num_leaves=num_leaves,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
        verbose: bool = False,
        **kwargs,
    ) -> "LightGBMModel":
        fit_params: dict = {}
        if verbose:
            fit_params["callbacks"] = None
        else:
            from lightgbm import log_evaluation
            fit_params["callbacks"] = [log_evaluation(period=-1)]

        if eval_set is not None and self.early_stopping_rounds is not None:
            from lightgbm import early_stopping
            if fit_params.get("callbacks") is None:
                fit_params["callbacks"] = []
            fit_params["callbacks"].append(early_stopping(self.early_stopping_rounds))
            fit_params["eval_set"] = eval_set
        elif eval_set is not None:
            fit_params["eval_set"] = eval_set

        fit_params.update(kwargs)
        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_

    @property
    def best_iteration(self) -> int:
        return self.model.best_iteration_

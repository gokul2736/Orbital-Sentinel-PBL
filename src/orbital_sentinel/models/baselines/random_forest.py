"""Random Forest regressor for collision risk prediction."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from orbital_sentinel.models import BaseModel


class RandomForestModel(BaseModel):

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 20,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "RandomForestModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_trees(self, X: np.ndarray) -> np.ndarray:
        """Return predictions from each individual tree. Shape: (n_trees, n_samples)."""
        return np.array([tree.predict(X) for tree in self.model.estimators_])

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_

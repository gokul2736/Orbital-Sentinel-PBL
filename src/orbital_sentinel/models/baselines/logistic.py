"""Ridge regression baseline for collision risk prediction.

File kept as logistic.py per project layout, but implements Ridge (linear)
regression since the target is continuous log10(collision probability).
"""

import numpy as np
from sklearn.linear_model import Ridge

from orbital_sentinel.models import BaseModel


class RidgeModel(BaseModel):

    def __init__(self, alpha: float = 1.0, random_state: int = 42, **kwargs):
        self.model = Ridge(alpha=alpha, random_state=random_state, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "RidgeModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        return self.model.coef_

    @property
    def intercept_(self) -> float:
        return self.model.intercept_

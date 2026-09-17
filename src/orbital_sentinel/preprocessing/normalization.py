from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from orbital_sentinel.config.settings import get_settings


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    settings = get_settings()
    exclude = set(
        settings.features.id_columns
        + settings.features.target_columns
        + settings.features.leakage_columns
    )
    return [col for col in df.columns if col not in exclude]


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def normalize(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X)

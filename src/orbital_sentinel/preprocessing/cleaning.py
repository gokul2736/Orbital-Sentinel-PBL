import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from orbital_sentinel.config.settings import get_settings


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    settings = get_settings()
    df = df.copy()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    all_nan_cols = df.columns[df.isna().all()]
    df.drop(columns=all_nan_cols, inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in settings.features.categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

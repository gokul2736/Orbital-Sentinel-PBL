import numpy as np
import pandas as pd

from orbital_sentinel.config.settings import get_settings


def assess_quality(df: pd.DataFrame) -> dict:
    settings = get_settings()
    floor_value = settings.target.floor_value

    total_rows = len(df)
    total_columns = len(df.columns)

    null_row_count = int(df.isnull().any(axis=1).sum())

    null_fractions = df.isnull().mean()
    columns_with_nulls = null_fractions[null_fractions > 0.5].index.tolist()

    risk_floor_pct = 0.0
    if "risk" in df.columns and total_rows > 0:
        risk_floor_pct = float((df["risk"] == floor_value).sum() / total_rows * 100)

    negative_miss_distance_count = 0
    if "miss_distance" in df.columns:
        negative_miss_distance_count = int((df["miss_distance"] <= 0).sum())

    negative_speed_count = 0
    if "relative_speed" in df.columns:
        negative_speed_count = int((df["relative_speed"] <= 0).sum())

    outlier_columns = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 2:
            continue
        mean = series.mean()
        std = series.std()
        if std == 0:
            continue
        if ((series - mean).abs() > 10 * std).any():
            outlier_columns.append(col)

    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "null_row_count": null_row_count,
        "columns_with_nulls": columns_with_nulls,
        "risk_floor_pct": risk_floor_pct,
        "negative_miss_distance_count": negative_miss_distance_count,
        "negative_speed_count": negative_speed_count,
        "outlier_columns": outlier_columns,
    }

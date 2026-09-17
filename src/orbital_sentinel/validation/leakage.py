import numpy as np
import pandas as pd

from orbital_sentinel.config.settings import get_settings

SAMPLE_SIZE = 10_000


def check_leakage(df: pd.DataFrame, feature_columns: list) -> list:
    settings = get_settings()
    leakage_cols = settings.features.leakage_columns
    target_col = settings.target.column
    warnings = []

    leaked_known = [col for col in leakage_cols if col in feature_columns]
    if leaked_known:
        warnings.append(
            f"Known leakage columns found in features: {leaked_known}"
        )

    risky_cols = [
        col for col in feature_columns
        if col != target_col and ("risk" in col.lower() or "max_risk" in col.lower())
    ]
    if risky_cols:
        warnings.append(
            f"Columns containing 'risk' found in features: {risky_cols}"
        )

    if target_col not in df.columns:
        return warnings

    target = df[target_col]
    if not pd.api.types.is_numeric_dtype(target):
        return warnings

    numeric_features = [
        col for col in feature_columns
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(df) > SAMPLE_SIZE:
        sample = df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        sample = df

    target_sample = sample[target_col]
    high_corr_cols = []
    for col in numeric_features:
        feature_sample = sample[col]
        valid = feature_sample.notna() & target_sample.notna()
        if valid.sum() < 10:
            continue
        corr = feature_sample[valid].corr(target_sample[valid])
        if abs(corr) > 0.95:
            high_corr_cols.append((col, round(corr, 4)))

    if high_corr_cols:
        details = ", ".join(f"{col} (r={corr})" for col, corr in high_corr_cols)
        warnings.append(
            f"Features with very high correlation (>0.95) to target: {details}"
        )

    return warnings

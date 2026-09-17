import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbital_sentinel.preprocessing.cleaning import clean_dataframe
from orbital_sentinel.preprocessing.normalization import get_feature_columns, fit_scaler, normalize
from orbital_sentinel.preprocessing.splitting import event_based_split
from orbital_sentinel.preprocessing import get_preprocessed_data


def test_clean_dataframe_replaces_inf_and_nan():
    df = pd.DataFrame({
        "event_id": [1, 2, 3, 4],
        "a": [1.0, np.inf, -np.inf, 4.0],
        "b": [np.nan, 2.0, 3.0, 4.0],
    })
    cleaned = clean_dataframe(df)
    assert not np.isinf(cleaned["a"]).any()
    assert not cleaned["b"].isna().any()


def test_clean_dataframe_drops_all_nan_columns():
    df = pd.DataFrame({
        "event_id": [1, 2],
        "good": [1.0, 2.0],
        "all_nan": [np.nan, np.nan],
    })
    cleaned = clean_dataframe(df)
    assert "all_nan" not in cleaned.columns
    assert "good" in cleaned.columns


def test_get_feature_columns_excludes_id_target_leakage():
    df = pd.DataFrame({
        "event_id": [1],
        "mission_id": [1],
        "risk": [-5.0],
        "max_risk_estimate": [1.0],
        "max_risk_scaling": [1.0],
        "miss_distance": [100.0],
        "relative_speed": [5.0],
    })
    feat_cols = get_feature_columns(df)
    assert "event_id" not in feat_cols
    assert "mission_id" not in feat_cols
    assert "risk" not in feat_cols
    assert "max_risk_estimate" not in feat_cols
    assert "max_risk_scaling" not in feat_cols
    assert "miss_distance" in feat_cols
    assert "relative_speed" in feat_cols


def test_event_based_split_keeps_event_in_same_split():
    df = pd.DataFrame({
        "event_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        "value": range(10),
    })
    train_df, val_df = event_based_split(df, test_size=0.5, random_state=42)
    train_events = set(train_df["event_id"].unique())
    val_events = set(val_df["event_id"].unique())
    assert train_events.isdisjoint(val_events), "Events leak across splits"
    assert len(train_df) + len(val_df) == len(df)


def test_get_preprocessed_data_returns_expected_keys():
    result = get_preprocessed_data()
    expected_keys = {
        "X_train", "X_val", "y_train", "y_val",
        "feature_names", "scaler", "train_df", "val_df",
    }
    assert set(result.keys()) == expected_keys
    assert result["X_train"].shape[0] > 0
    assert result["X_val"].shape[0] > 0
    assert len(result["y_train"]) == result["X_train"].shape[0]
    assert len(result["y_val"]) == result["X_val"].shape[0]


def test_fit_scaler_and_normalize():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = fit_scaler(X)
    X_norm = normalize(X, scaler)
    assert X_norm.shape == X.shape
    np.testing.assert_allclose(X_norm.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(X_norm.std(axis=0, ddof=0), 1.0, atol=1e-10)

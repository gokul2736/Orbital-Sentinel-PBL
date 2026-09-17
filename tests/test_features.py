import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbital_sentinel.ingestion.dataset_loader import load_esa_kelvins
from orbital_sentinel.features import engineer_all_features
from orbital_sentinel.features.static import engineer_static_features
from orbital_sentinel.features.orbital import engineer_orbital_features
from orbital_sentinel.features.covariance import engineer_covariance_features
from orbital_sentinel.features.temporal import engineer_temporal_features


@pytest.fixture
def sample_data():
    df = load_esa_kelvins(split="train")
    return df.head(100).copy()


def test_engineer_all_features_adds_columns(sample_data):
    original_cols = set(sample_data.columns)
    result = engineer_all_features(sample_data)
    new_cols = set(result.columns) - original_cols
    assert len(new_cols) > 0, "engineer_all_features should add new columns"
    assert len(result) == len(sample_data)


def test_engineer_static_features(sample_data):
    result = engineer_static_features(sample_data)
    assert "relative_position_magnitude" in result.columns
    assert "relative_velocity_magnitude" in result.columns
    assert "encounter_duration_proxy" in result.columns
    assert len(result) == len(sample_data)

    expected_mag = np.sqrt(
        sample_data["relative_position_r"] ** 2
        + sample_data["relative_position_t"] ** 2
        + sample_data["relative_position_n"] ** 2
    )
    np.testing.assert_allclose(
        result["relative_position_magnitude"].values,
        expected_mag.values,
        rtol=1e-10,
    )


def test_engineer_orbital_features(sample_data):
    result = engineer_orbital_features(sample_data)
    assert "t_orbital_period" in result.columns
    assert "t_mean_motion" in result.columns
    assert "t_perigee_alt" in result.columns
    assert "t_apogee_alt" in result.columns
    assert "sma_difference" in result.columns
    assert len(result) == len(sample_data)
    assert (result["t_orbital_period"] > 0).all()


def test_engineer_covariance_features(sample_data):
    result = engineer_covariance_features(sample_data)
    assert "t_position_uncertainty" in result.columns
    assert "c_position_uncertainty" in result.columns
    assert "combined_position_uncertainty" in result.columns
    assert len(result) == len(sample_data)


def test_engineer_temporal_features(sample_data):
    result = engineer_temporal_features(sample_data)
    assert "time_to_tca_hours" in result.columns
    assert "time_to_tca_squared" in result.columns
    assert "log_time_to_tca" in result.columns
    assert "is_close_approach" in result.columns
    assert "cdm_sequence_number" in result.columns
    assert len(result) == len(sample_data)

    np.testing.assert_allclose(
        result["time_to_tca_hours"].values,
        sample_data["time_to_tca"].values * 24.0,
        rtol=1e-10,
    )


def test_engineer_all_features_does_not_modify_input(sample_data):
    original_cols = list(sample_data.columns)
    original_len = len(sample_data)
    _ = engineer_all_features(sample_data)
    assert list(sample_data.columns) == original_cols
    assert len(sample_data) == original_len

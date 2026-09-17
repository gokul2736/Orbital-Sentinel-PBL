import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbital_sentinel.physics import verify_conjunction_physics
from orbital_sentinel.physics.collision_probability import (
    compute_collision_probability_2d,
    collision_probability_from_row,
)
from orbital_sentinel.physics.geometry import (
    compute_approach_angle,
    compute_encounter_duration,
    compute_relative_state,
    conjunction_geometry_summary,
)
from orbital_sentinel.physics.covariance import (
    build_position_covariance_rtn,
    combine_covariances,
    validate_covariance,
    extract_covariance_from_row,
    project_to_encounter_plane,
)
from orbital_sentinel.physics.propagation import (
    orbital_period,
    orbital_velocity,
    perigee_from_elements,
    apogee_from_elements,
    is_encounter_short,
)
from orbital_sentinel.ingestion.dataset_loader import load_esa_kelvins


@pytest.fixture
def real_row():
    df = load_esa_kelvins(split="train")
    return df.iloc[0].to_dict()


@pytest.fixture
def synthetic_row():
    return {
        "miss_distance": 0.5,
        "relative_speed": 10.0,
        "relative_position_r": 0.3,
        "relative_position_t": 0.2,
        "relative_position_n": 0.1,
        "relative_velocity_r": 2.0,
        "relative_velocity_t": 8.0,
        "relative_velocity_n": 1.0,
        "t_sigma_r": 0.05,
        "t_sigma_t": 0.1,
        "t_sigma_n": 0.03,
        "t_ct_r": 0.001,
        "t_cn_r": 0.0005,
        "t_cn_t": 0.0003,
        "c_sigma_r": 0.04,
        "c_sigma_t": 0.08,
        "c_sigma_n": 0.02,
        "c_ct_r": 0.0008,
        "c_cn_r": 0.0004,
        "c_cn_t": 0.0002,
        "t_j2k_sma": 7000.0,
        "t_j2k_ecc": 0.001,
        "c_j2k_sma": 7005.0,
        "c_j2k_ecc": 0.002,
        "mahalanobis_distance": 5.0,
    }


def test_verify_conjunction_physics_on_real_data(real_row):
    result = verify_conjunction_physics(real_row)
    assert "geometry" in result
    assert "covariance_valid" in result
    assert "analytic_pc" in result
    assert "log10_pc" in result
    assert "short_encounter_valid" in result
    assert "physical_consistency" in result
    assert isinstance(result["physical_consistency"], list)


def test_verify_conjunction_physics_on_synthetic(synthetic_row):
    result = verify_conjunction_physics(synthetic_row)
    assert result["geometry"]["miss_distance_km"] == pytest.approx(0.5, abs=0.01)
    assert result["geometry"]["relative_speed_km_s"] == pytest.approx(10.0, abs=0.1)
    assert isinstance(result["covariance_valid"], bool)


def test_compute_collision_probability_2d_basic():
    cov = np.array([[100.0, 0.0], [0.0, 100.0]])
    pc = compute_collision_probability_2d(
        miss_distance_m=0.0,
        combined_covariance_2d=cov,
        combined_radius_m=20.0,
    )
    assert 0.0 <= pc <= 1.0
    assert pc > 0.0


def test_compute_collision_probability_2d_large_miss():
    cov = np.array([[100.0, 0.0], [0.0, 100.0]])
    pc = compute_collision_probability_2d(
        miss_distance_m=100000.0,
        combined_covariance_2d=cov,
        combined_radius_m=20.0,
    )
    assert pc < 1e-10


def test_compute_collision_probability_2d_singular_cov():
    cov = np.array([[0.0, 0.0], [0.0, 0.0]])
    pc = compute_collision_probability_2d(
        miss_distance_m=10.0,
        combined_covariance_2d=cov,
        combined_radius_m=20.0,
    )
    assert pc == 0.0


def test_compute_approach_angle():
    pos = np.array([1.0, 0.0, 0.0])
    vel = np.array([1.0, 0.0, 0.0])
    angle = compute_approach_angle(pos, vel)
    assert angle == pytest.approx(0.0, abs=1e-10)

    vel_perp = np.array([0.0, 1.0, 0.0])
    angle_perp = compute_approach_angle(pos, vel_perp)
    assert angle_perp == pytest.approx(np.pi / 2, abs=1e-10)


def test_compute_encounter_duration():
    dur = compute_encounter_duration(miss_distance=1.0, relative_speed=10.0)
    assert dur == pytest.approx(10.0 / 10.0, abs=0.01)

    dur_zero_speed = compute_encounter_duration(miss_distance=1.0, relative_speed=0.0)
    assert dur_zero_speed == float("inf")


def test_conjunction_geometry_summary(synthetic_row):
    summary = conjunction_geometry_summary(synthetic_row)
    assert "miss_distance_km" in summary
    assert "approach_angle_deg" in summary
    assert "encounter_type" in summary
    assert summary["encounter_type"] in ("head-on", "overtaking", "crossing", "unknown")


def test_build_position_covariance_rtn():
    cov = build_position_covariance_rtn(
        sigma_r=1.0, sigma_t=2.0, sigma_n=0.5,
        ct_r=0.1, cn_r=0.05, cn_t=0.02,
    )
    assert cov.shape == (3, 3)
    np.testing.assert_allclose(cov, cov.T)
    assert cov[0, 0] == pytest.approx(1.0)
    assert cov[1, 1] == pytest.approx(4.0)


def test_validate_covariance_valid():
    cov = np.diag([1.0, 2.0, 0.5])
    warnings = validate_covariance(cov)
    assert warnings == []


def test_validate_covariance_negative_eigenvalue():
    cov = np.array([[1.0, 5.0, 0.0], [5.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    warnings = validate_covariance(cov)
    assert any("Not positive semi-definite" in w or "Correlation" in w for w in warnings)


def test_orbital_period():
    period = orbital_period(6778.0)
    assert 5000 < period < 6000

    with pytest.raises(ValueError):
        orbital_period(-100.0)


def test_orbital_velocity():
    v = orbital_velocity(6778.0, 6778.0)
    assert 7.0 < v < 8.0

    with pytest.raises(ValueError):
        orbital_velocity(-100.0, 6778.0)


def test_perigee_apogee_from_elements():
    perigee = perigee_from_elements(7000.0, 0.01)
    apogee = apogee_from_elements(7000.0, 0.01)
    assert perigee < apogee
    assert perigee > 0
    assert apogee > 0


def test_is_encounter_short():
    assert is_encounter_short(1.0, 5400.0) is True
    assert is_encounter_short(100.0, 5400.0) is False
    assert is_encounter_short(1.0, 0.0) is False

"""Orbital mechanics utilities for conjunction analysis.

These are simple Keplerian utilities, NOT a full orbital propagator.
All assume two-body dynamics with Earth as the central body.
"""

import numpy as np

MU_EARTH = 398600.4418  # km^3/s^2, Earth gravitational parameter (WGS-84)
R_EARTH = 6378.137  # km, Earth equatorial radius (WGS-84)


def orbital_period(semi_major_axis_km: float) -> float:
    """Compute orbital period in seconds using Kepler's third law.

    T = 2 * pi * sqrt(a^3 / mu)

    Assumes: two-body Keplerian orbit, no perturbations (J2, drag, etc.).
    """
    if semi_major_axis_km <= 0:
        raise ValueError(f"Semi-major axis must be positive, got {semi_major_axis_km} km")
    return 2.0 * np.pi * np.sqrt(semi_major_axis_km**3 / MU_EARTH)


def orbital_velocity(semi_major_axis_km: float, radius_km: float) -> float:
    """Compute orbital velocity via the vis-viva equation.

    v = sqrt(mu * (2/r - 1/a))

    Returns velocity in km/s. Assumes Keplerian orbit.
    """
    if semi_major_axis_km <= 0:
        raise ValueError(f"Semi-major axis must be positive, got {semi_major_axis_km} km")
    if radius_km <= 0:
        raise ValueError(f"Radius must be positive, got {radius_km} km")
    return np.sqrt(MU_EARTH * (2.0 / radius_km - 1.0 / semi_major_axis_km))


def is_encounter_short(encounter_duration_s: float, orbital_period_s: float) -> bool:
    """Check if the short-encounter assumption is valid.

    The Foster/Akella-Alfriend 2D Pc formula assumes the encounter duration
    is much shorter than the orbital period. A common threshold is 1% of
    the orbital period.
    """
    if orbital_period_s <= 0:
        return False
    return encounter_duration_s < 0.01 * orbital_period_s


def perigee_from_elements(sma_km: float, ecc: float) -> float:
    """Compute perigee altitude in km above Earth's surface.

    perigee_altitude = a * (1 - e) - R_Earth
    """
    return sma_km * (1.0 - ecc) - R_EARTH


def apogee_from_elements(sma_km: float, ecc: float) -> float:
    """Compute apogee altitude in km above Earth's surface.

    apogee_altitude = a * (1 + e) - R_Earth
    """
    return sma_km * (1.0 + ecc) - R_EARTH

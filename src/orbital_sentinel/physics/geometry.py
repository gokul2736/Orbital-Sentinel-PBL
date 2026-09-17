"""Conjunction geometry computations in the RTN (Radial-Transverse-Normal) frame.

RTN frame convention:
  R (Radial)     -- from Earth center through the target spacecraft
  T (Transverse) -- in the orbital plane, perpendicular to R, in velocity direction
  N (Normal)     -- completes the right-hand system (orbit normal)

All inputs from the ESA Kelvins dataset use this convention.
"""

import numpy as np


def compute_relative_state(row: dict) -> dict:
    """Extract and validate relative position/velocity in RTN frame from a data row.

    Expects keys: miss_distance, relative_speed,
    relative_position_r/t/n (km), relative_velocity_r/t/n (km/s).

    Returns dict with miss_distance, relative_speed, position_rtn (3-vector),
    velocity_rtn (3-vector), and any warnings.
    """
    warnings = []

    pos_keys = ["relative_position_r", "relative_position_t", "relative_position_n"]
    vel_keys = ["relative_velocity_r", "relative_velocity_t", "relative_velocity_n"]

    position_rtn = np.array([_safe_float(row.get(k), k, warnings) for k in pos_keys])
    velocity_rtn = np.array([_safe_float(row.get(k), k, warnings) for k in vel_keys])

    miss_distance = _safe_float(row.get("miss_distance"), "miss_distance", warnings)
    relative_speed = _safe_float(row.get("relative_speed"), "relative_speed", warnings)

    computed_miss = np.linalg.norm(position_rtn)
    if miss_distance > 0 and computed_miss > 0:
        rel_diff = abs(computed_miss - miss_distance) / max(miss_distance, 1e-12)
        if rel_diff > 0.01:
            warnings.append(
                f"miss_distance ({miss_distance:.4f} km) differs from "
                f"norm(position_rtn) ({computed_miss:.4f} km) by {rel_diff*100:.1f}%"
            )

    return {
        "miss_distance": miss_distance,
        "relative_speed": relative_speed,
        "position_rtn": position_rtn,
        "velocity_rtn": velocity_rtn,
        "warnings": warnings,
    }


def conjunction_geometry_summary(row: dict) -> dict:
    """Compute geometric properties of a conjunction event.

    Returns approach angle, miss distance components, relative velocity direction,
    and encounter classification (head-on / overtaking / crossing).
    """
    state = compute_relative_state(row)
    pos = state["position_rtn"]
    vel = state["velocity_rtn"]

    approach_angle = compute_approach_angle(pos, vel)
    encounter_type = _classify_encounter(vel)
    speed = state["relative_speed"] if state["relative_speed"] > 0 else np.linalg.norm(vel)
    encounter_duration = compute_encounter_duration(state["miss_distance"], speed)

    return {
        "miss_distance_km": state["miss_distance"],
        "miss_distance_r_km": pos[0],
        "miss_distance_t_km": pos[1],
        "miss_distance_n_km": pos[2],
        "relative_speed_km_s": speed,
        "approach_angle_rad": approach_angle,
        "approach_angle_deg": np.degrees(approach_angle),
        "encounter_type": encounter_type,
        "encounter_duration_s": encounter_duration,
        "warnings": state["warnings"],
    }


def compute_approach_angle(rel_pos_rtn: np.ndarray, rel_vel_rtn: np.ndarray) -> float:
    """Angle between relative position and velocity vectors (radians).

    Returns value in [0, pi]. Returns 0.0 if either vector has zero magnitude.
    """
    pos_norm = np.linalg.norm(rel_pos_rtn)
    vel_norm = np.linalg.norm(rel_vel_rtn)

    if pos_norm < 1e-15 or vel_norm < 1e-15:
        return 0.0

    cos_angle = np.dot(rel_pos_rtn, rel_vel_rtn) / (pos_norm * vel_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def compute_encounter_duration(miss_distance: float, relative_speed: float) -> float:
    """Approximate encounter duration assuming a linear fly-by (seconds).

    Uses a characteristic length scale of 10x the miss distance (or a minimum
    of 1 km) divided by the relative speed. This gives the time over which
    the objects are "near" closest approach.

    Assumes: relative velocity is approximately constant during the encounter.
    """
    if relative_speed <= 0:
        return float("inf")

    characteristic_length_km = max(miss_distance * 10.0, 1.0)
    return characteristic_length_km / relative_speed


def _classify_encounter(vel_rtn: np.ndarray) -> str:
    """Classify encounter geometry based on relative velocity direction in RTN.

    - head-on: dominant along-track component with high relative speed (>5 km/s)
    - overtaking: dominant along-track component with low relative speed
    - crossing: dominant radial or normal component
    """
    abs_vel = np.abs(vel_rtn)
    total = np.sum(abs_vel)
    if total < 1e-15:
        return "unknown"

    fractions = abs_vel / total
    _r_frac, t_frac, _n_frac = fractions

    if t_frac > 0.5:
        speed = np.linalg.norm(vel_rtn)
        if speed > 5.0:
            return "head-on"
        return "overtaking"

    return "crossing"


def _safe_float(value, name: str, warnings: list) -> float:
    """Convert a value to float, appending a warning if NaN or missing."""
    if value is None:
        warnings.append(f"{name} is missing")
        return 0.0
    f = float(value)
    if np.isnan(f):
        warnings.append(f"{name} is NaN")
        return 0.0
    return f

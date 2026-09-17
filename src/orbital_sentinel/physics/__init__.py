"""Physics verification layer for conjunction analysis.

Provides tools for computing and validating collision probabilities using
the standard 2D short-encounter approximation (Foster 1992 / Akella-Alfriend).
"""

import numpy as np

from .collision_probability import collision_probability_from_row, compute_collision_probability_2d
from .covariance import (
    build_position_covariance_rtn,
    combine_covariances,
    extract_covariance_from_row,
    project_miss_to_encounter_plane,
    project_to_encounter_plane,
    validate_covariance,
)
from .geometry import (
    compute_approach_angle,
    compute_encounter_duration,
    compute_relative_state,
    conjunction_geometry_summary,
)
from .monte_carlo import monte_carlo_collision_probability, verify_analytic_with_monte_carlo
from .propagation import (
    MU_EARTH,
    R_EARTH,
    apogee_from_elements,
    is_encounter_short,
    orbital_period,
    orbital_velocity,
    perigee_from_elements,
)


def verify_conjunction_physics(row: dict) -> dict:
    """Complete physics verification for a conjunction event.

    Runs all physics checks on a single conjunction data row and returns
    a comprehensive result dictionary.

    Parameters
    ----------
    row : dict
        A data row from the ESA Kelvins dataset (or any dict with the
        expected column names).

    Returns
    -------
    dict with keys:
        geometry : dict - approach angle, encounter type, miss components
        covariance_valid : bool - whether covariance matrices pass validation
        covariance_warnings : list of str
        analytic_pc : float - 2D collision probability
        log10_pc : float - log10 of Pc
        mc_verification : dict or None - Monte Carlo cross-check (if feasible)
        short_encounter_valid : bool - whether the short-encounter assumption holds
        physical_consistency : list of dict - checks passed/failed with details
    """
    warnings_all = []
    checks = []

    # --- Geometry ---
    geometry = conjunction_geometry_summary(row)
    warnings_all.extend(geometry.get("warnings", []))

    # --- Covariance validation ---
    cov_warnings = []
    try:
        cov_t = extract_covariance_from_row(row, "t")
        cov_c = extract_covariance_from_row(row, "c")
        cov_warnings.extend([f"Target: {w}" for w in validate_covariance(cov_t)])
        cov_warnings.extend([f"Chaser: {w}" for w in validate_covariance(cov_c)])
    except (KeyError, TypeError, ValueError) as e:
        cov_warnings.append(f"Covariance extraction failed: {e}")

    covariance_valid = len(cov_warnings) == 0

    # --- Collision probability ---
    pc_result = collision_probability_from_row(row)
    warnings_all.extend(pc_result.get("warnings", []))

    # --- Short-encounter check ---
    short_encounter_valid = False
    sma = row.get("t_j2k_sma")
    if sma is not None and float(sma) > 0:
        try:
            period_s = orbital_period(float(sma))
            enc_dur = geometry["encounter_duration_s"]
            short_encounter_valid = is_encounter_short(enc_dur, period_s)
            checks.append({
                "check": "short_encounter_assumption",
                "passed": short_encounter_valid,
                "detail": (
                    f"encounter_duration={enc_dur:.1f}s, "
                    f"orbital_period={period_s:.0f}s, "
                    f"ratio={enc_dur/period_s:.6f}"
                ),
            })
        except ValueError:
            checks.append({
                "check": "short_encounter_assumption",
                "passed": False,
                "detail": "Could not compute orbital period",
            })
    else:
        checks.append({
            "check": "short_encounter_assumption",
            "passed": False,
            "detail": "Semi-major axis not available",
        })

    # --- Miss distance vs sigma check ---
    if pc_result["is_valid"] and not np.isnan(pc_result["combined_sigma_km"]):
        sigma_km = pc_result["combined_sigma_km"]
        md_km = geometry["miss_distance_km"]
        if sigma_km > 0:
            md_sigma_ratio = md_km / sigma_km
            checks.append({
                "check": "miss_distance_in_sigma",
                "passed": True,
                "detail": f"miss_distance={md_km:.4f} km = {md_sigma_ratio:.1f} sigma",
            })

    # --- Mahalanobis distance cross-check ---
    maha = row.get("mahalanobis_distance")
    if maha is not None:
        try:
            maha_val = float(maha)
            checks.append({
                "check": "mahalanobis_distance",
                "passed": maha_val < 100,
                "detail": f"mahalanobis_distance={maha_val:.2f}",
            })
        except (TypeError, ValueError):
            pass

    # --- Orbital altitude check ---
    t_sma = row.get("t_j2k_sma")
    t_ecc = row.get("t_j2k_ecc")
    if t_sma is not None and t_ecc is not None:
        try:
            t_sma_f = float(t_sma)
            t_ecc_f = float(t_ecc)
            perigee = perigee_from_elements(t_sma_f, t_ecc_f)
            apogee = apogee_from_elements(t_sma_f, t_ecc_f)
            reasonable = perigee > 100 and apogee < 100000
            checks.append({
                "check": "target_orbit_reasonable",
                "passed": reasonable,
                "detail": f"perigee={perigee:.0f} km, apogee={apogee:.0f} km",
            })
        except (TypeError, ValueError):
            pass

    # --- Monte Carlo verification (only if Pc is large enough to be sampled) ---
    mc_verification = None
    if pc_result["is_valid"] and pc_result["pc"] > 1e-7:
        try:
            state = compute_relative_state(row)
            vel_rtn = state["velocity_rtn"]
            miss_rtn = state["position_rtn"]

            combined_3d = combine_covariances(
                extract_covariance_from_row(row, "t"),
                extract_covariance_from_row(row, "c"),
            )
            cov_2d_km2 = project_to_encounter_plane(combined_3d, vel_rtn)
            miss_2d_km = project_miss_to_encounter_plane(miss_rtn, vel_rtn)

            cov_2d_m2 = cov_2d_km2 * 1e6
            miss_2d_m = miss_2d_km * 1e3

            mc_verification = verify_analytic_with_monte_carlo(
                miss_vector=miss_2d_m,
                combined_covariance_2d=cov_2d_m2,
                combined_radius_m=20.0,
                n_samples=10000,
            )
            checks.append({
                "check": "mc_analytic_agreement",
                "passed": mc_verification["agreement"],
                "detail": (
                    f"analytic={mc_verification['analytic_pc']:.2e}, "
                    f"mc={mc_verification['mc_pc']:.2e}"
                ),
            })
        except Exception as e:
            mc_verification = {"error": str(e)}

    return {
        "geometry": {
            "miss_distance_km": geometry["miss_distance_km"],
            "miss_r_km": geometry["miss_distance_r_km"],
            "miss_t_km": geometry["miss_distance_t_km"],
            "miss_n_km": geometry["miss_distance_n_km"],
            "relative_speed_km_s": geometry["relative_speed_km_s"],
            "approach_angle_deg": geometry["approach_angle_deg"],
            "encounter_type": geometry["encounter_type"],
            "encounter_duration_s": geometry["encounter_duration_s"],
        },
        "covariance_valid": covariance_valid,
        "covariance_warnings": cov_warnings,
        "analytic_pc": pc_result["pc"],
        "log10_pc": pc_result["log10_pc"],
        "mc_verification": mc_verification,
        "short_encounter_valid": short_encounter_valid,
        "physical_consistency": checks,
    }

"""Collision probability computation using the 2D short-encounter approximation.

Implements the Foster (1992) / Akella-Alfriend short-encounter Pc formula,
which is the standard method used by NASA, ESA, and the 18th Space Defense
Squadron for operational conjunction assessment.

Assumptions and Limitations
---------------------------
1. SHORT-ENCOUNTER: The encounter duration is much shorter than the orbital
   period (typically < 1%). This means relative velocity is approximately
   constant during the closest approach, and the relative trajectory is
   a straight line through the combined uncertainty ellipsoid.

2. HARD-BODY: Both objects are modeled as spheres with a combined hard-body
   radius R = r_target + r_chaser. Default 20 m covers most active spacecraft
   plus debris. Real objects are not spherical; this is conservative.

3. GAUSSIAN UNCERTAINTY: Position errors are assumed to follow a multivariate
   normal distribution. This is reasonable for well-tracked objects but
   breaks down for long propagation times or sparse tracking.

4. FRAME: All computations assume the encounter is described in the target's
   RTN frame, projected onto the encounter (B-plane) perpendicular to the
   relative velocity.

5. UNITS: External interface in km; internal conversion to meters for the
   hard-body radius comparison. The covariance is converted km^2 -> m^2.

When these assumptions are violated, the 2D Pc is unreliable:
- Slow encounters (e.g., co-orbiting objects): use 3D methods or time-dependent Pc
- Very large miss distances (>> 3*sigma): Pc is effectively zero, formula is fine
  but numerically may underflow
- Singular or near-singular covariance: degenerate case, flagged with warnings

References
----------
- Foster, J.L. (1992), "The Analytic Basis for Debris Avoidance Operations"
- Akella, M.R. and Alfriend, K.T. (2000), "Probability of Collision Between
  Space Objects", Journal of Guidance, Control, and Dynamics
- Alfano, S. (2005), "A Numerical Implementation of Spherical Object
  Collision Probability", Journal of the Astronautical Sciences
"""

import numpy as np
from scipy.stats import multivariate_normal

from .covariance import (
    combine_covariances,
    extract_covariance_from_row,
    project_miss_to_encounter_plane,
    project_to_encounter_plane,
    validate_covariance,
)
from .geometry import compute_relative_state


def compute_collision_probability_2d(
    miss_distance_m: float,
    combined_covariance_2d: np.ndarray,
    combined_radius_m: float = 20.0,
    miss_vector_2d: np.ndarray = None,
) -> float:
    """Compute 2D collision probability using the short-encounter approximation.

    This evaluates the probability that the miss vector falls within the
    hard-body disk of radius R, given a 2D Gaussian distribution:

        Pc = integral over disk(R) of N(mu, C) dA

    where mu is the miss vector projected onto the encounter plane and C is
    the 2x2 projected covariance.

    For the general case (non-zero miss, correlated covariance), we use
    scipy.stats.multivariate_normal to numerically integrate via the CDF
    evaluated on a fine grid covering the hard-body disk.

    For the special case of zero miss with diagonal covariance:
        Pc = 1 - exp(-R^2 / (2 * sigma_x * sigma_y))

    Parameters
    ----------
    miss_distance_m : float
        Scalar miss distance in meters (used only if miss_vector_2d is None).
    combined_covariance_2d : np.ndarray
        2x2 covariance matrix in the encounter plane, in m^2.
    combined_radius_m : float
        Combined hard-body radius of both objects in meters.
    miss_vector_2d : np.ndarray, optional
        2D miss vector in the encounter plane (meters). If None, the miss
        is placed along the first axis: [miss_distance_m, 0].

    Returns
    -------
    float
        Collision probability (dimensionless, in [0, 1]).
    """
    if miss_vector_2d is None:
        miss_vector_2d = np.array([miss_distance_m, 0.0])

    det = np.linalg.det(combined_covariance_2d)
    if det <= 0:
        return 0.0

    sigma_x = np.sqrt(combined_covariance_2d[0, 0])
    sigma_y = np.sqrt(combined_covariance_2d[1, 1])

    if sigma_x <= 0 or sigma_y <= 0:
        return 0.0

    # Use the Alfano grid integration approach: evaluate the 2D Gaussian PDF
    # over a fine grid covering the hard-body disk, then sum.
    n_grid = 200
    r = combined_radius_m
    x = np.linspace(-r, r, n_grid)
    y = np.linspace(-r, r, n_grid)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    xx, yy = np.meshgrid(x, y)
    disk_mask = xx**2 + yy**2 <= r**2

    rv = multivariate_normal(mean=miss_vector_2d, cov=combined_covariance_2d)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    pdf_values = rv.pdf(points).reshape(xx.shape)

    pc = float(np.sum(pdf_values[disk_mask]) * dx * dy)

    return np.clip(pc, 0.0, 1.0)


def _compute_pc_foster_analytic(
    miss_vector_2d: np.ndarray,
    combined_covariance_2d: np.ndarray,
    combined_radius_m: float,
) -> float:
    """Foster/Akella-Alfriend analytic approximation for near-circular hard body.

    Pc = (R^2) / (2 * sigma_x' * sigma_y') * exp(-0.5 * d'^T C'^-1 d')

    where primed quantities are in the principal axes of the 2D covariance
    (diagonalized). This is a first-order approximation valid when R << sigma.

    This is used as a fast cross-check, not as the primary computation.
    """
    eigvals, eigvecs = np.linalg.eigh(combined_covariance_2d)
    if np.any(eigvals <= 0):
        return 0.0

    sigma_x = np.sqrt(eigvals[0])
    sigma_y = np.sqrt(eigvals[1])

    # Rotate miss vector into principal axes
    d_prime = eigvecs.T @ miss_vector_2d

    u = d_prime[0]**2 / (2.0 * eigvals[0]) + d_prime[1]**2 / (2.0 * eigvals[1])

    pc = (combined_radius_m**2 / (2.0 * sigma_x * sigma_y)) * np.exp(-u)

    return float(np.clip(pc, 0.0, 1.0))


def collision_probability_from_row(
    row: dict, combined_radius_m: float = 20.0
) -> dict:
    """End-to-end collision probability computation from a dataset row.

    Steps:
    1. Extract relative state (position and velocity in RTN)
    2. Build and combine 3x3 covariance matrices for target and chaser
    3. Project to 2D encounter plane
    4. Compute 2D collision probability

    Parameters
    ----------
    row : dict
        Data row from the ESA Kelvins dataset.
    combined_radius_m : float
        Combined hard-body radius of both objects (meters).

    Returns
    -------
    dict with keys:
        pc : float - collision probability
        log10_pc : float - log10 of Pc (or -inf if Pc=0)
        miss_distance_km : float
        combined_sigma_km : float - sqrt of mean eigenvalue of combined 2D cov
        is_valid : bool - whether computation succeeded without critical warnings
        warnings : list of str
        foster_analytic_pc : float - Foster analytic cross-check
    """
    warnings = []

    state = compute_relative_state(row)
    warnings.extend(state["warnings"])

    miss_rtn = state["position_rtn"]
    vel_rtn = state["velocity_rtn"]

    has_velocity = np.linalg.norm(vel_rtn) > 1e-12
    if not has_velocity:
        warnings.append("Relative velocity is zero or missing; cannot define encounter plane")
        return _invalid_result(state["miss_distance"], warnings)

    try:
        cov_t = extract_covariance_from_row(row, "t")
        cov_c = extract_covariance_from_row(row, "c")
    except (KeyError, TypeError, ValueError) as e:
        warnings.append(f"Failed to extract covariance: {e}")
        return _invalid_result(state["miss_distance"], warnings)

    cov_warnings_t = validate_covariance(cov_t)
    cov_warnings_c = validate_covariance(cov_c)
    for w in cov_warnings_t:
        warnings.append(f"Target covariance: {w}")
    for w in cov_warnings_c:
        warnings.append(f"Chaser covariance: {w}")

    combined_3d = combine_covariances(cov_t, cov_c)

    has_critical = any(
        "Not positive semi-definite" in w or "NaN" in w for w in warnings
    )
    if has_critical:
        return _invalid_result(state["miss_distance"], warnings)

    try:
        cov_2d_km2 = project_to_encounter_plane(combined_3d, vel_rtn)
        miss_2d_km = project_miss_to_encounter_plane(miss_rtn, vel_rtn)
    except ValueError as e:
        warnings.append(str(e))
        return _invalid_result(state["miss_distance"], warnings)

    # Convert km -> m for Pc computation
    cov_2d_m2 = cov_2d_km2 * 1e6  # km^2 -> m^2
    miss_2d_m = miss_2d_km * 1e3   # km -> m

    miss_dist_m = np.linalg.norm(miss_2d_m)
    pc = compute_collision_probability_2d(
        miss_distance_m=miss_dist_m,
        combined_covariance_2d=cov_2d_m2,
        combined_radius_m=combined_radius_m,
        miss_vector_2d=miss_2d_m,
    )

    foster_pc = _compute_pc_foster_analytic(miss_2d_m, cov_2d_m2, combined_radius_m)

    eigvals_2d = np.linalg.eigvalsh(cov_2d_km2)
    combined_sigma_km = float(np.sqrt(np.mean(eigvals_2d)))

    log10_pc = float(np.log10(pc)) if pc > 0 else float("-inf")

    return {
        "pc": pc,
        "log10_pc": log10_pc,
        "miss_distance_km": state["miss_distance"],
        "combined_sigma_km": combined_sigma_km,
        "is_valid": True,
        "warnings": warnings,
        "foster_analytic_pc": foster_pc,
    }


def _invalid_result(miss_distance: float, warnings: list) -> dict:
    """Return a result dict for cases where Pc could not be computed."""
    return {
        "pc": float("nan"),
        "log10_pc": float("nan"),
        "miss_distance_km": miss_distance,
        "combined_sigma_km": float("nan"),
        "is_valid": False,
        "warnings": warnings,
        "foster_analytic_pc": float("nan"),
    }

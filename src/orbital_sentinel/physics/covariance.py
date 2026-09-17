"""Covariance matrix construction and manipulation for conjunction analysis.

Covariance data in the ESA Kelvins dataset follows the CCSDS Conjunction Data
Message (CDM) convention. Position covariance is given in the RTN frame as:
  - Diagonal: sigma_r^2, sigma_t^2, sigma_n^2
  - Off-diagonal: C(T,R), C(N,R), C(N,T)

The 3x3 position covariance matrix in RTN is symmetric:
    | sigma_r^2   C(T,R)      C(N,R)   |
    | C(T,R)      sigma_t^2   C(N,T)   |
    | C(N,R)      C(N,T)      sigma_n^2|
"""

from typing import List

import numpy as np


def build_position_covariance_rtn(
    sigma_r: float,
    sigma_t: float,
    sigma_n: float,
    ct_r: float,
    cn_r: float,
    cn_t: float,
) -> np.ndarray:
    """Build a 3x3 position covariance matrix in RTN frame from CDM elements.

    Parameters
    ----------
    sigma_r, sigma_t, sigma_n : float
        Position standard deviations in R, T, N directions (km).
    ct_r : float
        Covariance cross-term C(T,R) in km^2.
    cn_r : float
        Covariance cross-term C(N,R) in km^2.
    cn_t : float
        Covariance cross-term C(N,T) in km^2.

    Returns
    -------
    np.ndarray
        3x3 symmetric positive (semi-)definite covariance matrix in km^2.
    """
    cov = np.array([
        [sigma_r**2, ct_r, cn_r],
        [ct_r, sigma_t**2, cn_t],
        [cn_r, cn_t, sigma_n**2],
    ])
    return cov


def combine_covariances(cov_target: np.ndarray, cov_chaser: np.ndarray) -> np.ndarray:
    """Compute the combined covariance for relative position.

    For independent objects, the covariance of the relative position is
    the sum of the individual covariances: C_combined = C_target + C_chaser.

    Assumes: both covariances are expressed in the same frame (RTN of the
    target, as is standard in CDM data). The chaser covariance should already
    be rotated into the target RTN frame before calling this function. In the
    ESA Kelvins dataset, both are provided in the respective object's RTN frame,
    which is a common simplification -- the rotation effect is small for
    near-coplanar, near-circular orbits.
    """
    return cov_target + cov_chaser


def project_to_encounter_plane(
    combined_cov: np.ndarray, rel_vel_rtn: np.ndarray
) -> np.ndarray:
    """Project a 3D covariance onto the 2D encounter plane.

    The encounter plane is perpendicular to the relative velocity vector.
    We construct an orthonormal basis for this plane and project the 3D
    covariance via: C_2D = P @ C_3D @ P^T, where P is the 2x3 projection
    matrix whose rows span the encounter plane.

    Parameters
    ----------
    combined_cov : np.ndarray
        3x3 combined position covariance in RTN frame (km^2).
    rel_vel_rtn : np.ndarray
        Relative velocity vector in RTN frame (km/s).

    Returns
    -------
    np.ndarray
        2x2 covariance matrix in the encounter plane (km^2).

    Raises
    ------
    ValueError
        If relative velocity is near-zero (encounter plane undefined).
    """
    vel_norm = np.linalg.norm(rel_vel_rtn)
    if vel_norm < 1e-12:
        raise ValueError(
            "Relative velocity is near-zero; encounter plane is undefined. "
            "The short-encounter model does not apply."
        )

    v_hat = rel_vel_rtn / vel_norm

    # Build orthonormal basis for the plane perpendicular to v_hat.
    # Choose the cardinal axis least aligned with v_hat to avoid near-parallel issues.
    abs_v = np.abs(v_hat)
    min_idx = int(np.argmin(abs_v))
    seed = np.zeros(3)
    seed[min_idx] = 1.0

    e1 = np.cross(v_hat, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(v_hat, e1)
    e2 /= np.linalg.norm(e2)

    projection = np.vstack([e1, e2])  # 2x3
    cov_2d = projection @ combined_cov @ projection.T

    # Enforce exact symmetry (numerical hygiene)
    cov_2d = 0.5 * (cov_2d + cov_2d.T)
    return cov_2d


def project_miss_to_encounter_plane(
    miss_rtn: np.ndarray, rel_vel_rtn: np.ndarray
) -> np.ndarray:
    """Project the 3D miss vector onto the 2D encounter plane.

    Uses the same projection basis as project_to_encounter_plane for consistency.

    Returns
    -------
    np.ndarray
        2-element miss vector in the encounter plane (km).
    """
    vel_norm = np.linalg.norm(rel_vel_rtn)
    if vel_norm < 1e-12:
        raise ValueError("Relative velocity is near-zero; encounter plane is undefined.")

    v_hat = rel_vel_rtn / vel_norm
    abs_v = np.abs(v_hat)
    min_idx = int(np.argmin(abs_v))
    seed = np.zeros(3)
    seed[min_idx] = 1.0

    e1 = np.cross(v_hat, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(v_hat, e1)
    e2 /= np.linalg.norm(e2)

    return np.array([np.dot(e1, miss_rtn), np.dot(e2, miss_rtn)])


def validate_covariance(cov: np.ndarray) -> List[str]:
    """Validate a covariance matrix for physical reasonableness.

    Checks:
    1. Symmetry
    2. Positive semi-definiteness (all eigenvalues >= 0)
    3. Diagonal elements are non-negative
    4. Correlation coefficients in [-1, 1]
    5. Eigenvalue spread (condition number) -- warns if poorly conditioned

    Returns a list of warning strings (empty if all checks pass).
    """
    warnings = []

    if cov.shape[0] != cov.shape[1]:
        warnings.append(f"Covariance is not square: shape {cov.shape}")
        return warnings

    n = cov.shape[0]

    if np.any(np.isnan(cov)):
        warnings.append("Covariance contains NaN values")
        return warnings

    asym = np.max(np.abs(cov - cov.T))
    if asym > 1e-10:
        warnings.append(f"Covariance is not symmetric: max asymmetry {asym:.2e}")

    for i in range(n):
        if cov[i, i] < 0:
            warnings.append(f"Negative diagonal element cov[{i},{i}] = {cov[i, i]:.4e}")

    eigvals = np.linalg.eigvalsh(cov)
    min_eig = float(np.min(eigvals))
    if min_eig < -1e-10:
        warnings.append(f"Not positive semi-definite: min eigenvalue = {min_eig:.4e}")

    for i in range(n):
        for j in range(i + 1, n):
            si = np.sqrt(max(cov[i, i], 0))
            sj = np.sqrt(max(cov[j, j], 0))
            if si > 0 and sj > 0:
                rho = cov[i, j] / (si * sj)
                if abs(rho) > 1.0 + 1e-6:
                    warnings.append(
                        f"Correlation rho[{i},{j}] = {rho:.4f} is outside [-1, 1]"
                    )

    max_eig = float(np.max(eigvals))
    if min_eig > 0 and max_eig / min_eig > 1e10:
        warnings.append(
            f"Poorly conditioned: condition number = {max_eig / min_eig:.2e}"
        )

    return warnings


def extract_covariance_from_row(row: dict, prefix: str) -> np.ndarray:
    """Build a 3x3 position covariance from a dataset row for a given object.

    Parameters
    ----------
    row : dict
        Data row from the ESA Kelvins dataset.
    prefix : str
        Object prefix, either 't' (target) or 'c' (chaser).

    Returns
    -------
    np.ndarray
        3x3 position covariance in RTN frame (km^2).
    """
    sigma_r = float(row.get(f"{prefix}_sigma_r", 0))
    sigma_t = float(row.get(f"{prefix}_sigma_t", 0))
    sigma_n = float(row.get(f"{prefix}_sigma_n", 0))
    ct_r = float(row.get(f"{prefix}_ct_r", 0))
    cn_r = float(row.get(f"{prefix}_cn_r", 0))
    cn_t = float(row.get(f"{prefix}_cn_t", 0))

    return build_position_covariance_rtn(sigma_r, sigma_t, sigma_n, ct_r, cn_r, cn_t)

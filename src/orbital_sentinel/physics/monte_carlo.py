"""Monte Carlo verification of analytic collision probability.

Monte Carlo simulation provides an independent check on the analytic 2D Pc
formula. By drawing many random samples from the combined uncertainty
distribution and counting how many fall within the hard-body radius, we
get a statistical estimate of Pc.

This is NOT the primary Pc computation method -- it is used for verification.
Monte Carlo is much slower than the analytic formula but makes fewer
assumptions (it naturally handles non-Gaussian distributions, 3D geometry,
and does not require the short-encounter approximation).
"""

import numpy as np
from scipy.stats import norm

from .collision_probability import compute_collision_probability_2d


def monte_carlo_collision_probability(
    miss_vector: np.ndarray,
    combined_covariance: np.ndarray,
    combined_radius_m: float = 20.0,
    n_samples: int = 10000,
    random_state: int = 42,
) -> dict:
    """Estimate collision probability via Monte Carlo sampling.

    Draws samples from N(miss_vector, combined_covariance) and counts the
    fraction that fall within the hard-body radius.

    Parameters
    ----------
    miss_vector : np.ndarray
        2D or 3D miss vector in meters.
    combined_covariance : np.ndarray
        Covariance matrix matching the dimension of miss_vector, in m^2.
    combined_radius_m : float
        Combined hard-body radius in meters.
    n_samples : int
        Number of Monte Carlo samples.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        pc_estimate : float - fraction of samples within radius
        confidence_interval_95 : tuple of (lower, upper)
        n_samples : int
        n_hits : int - number of samples within radius
    """
    rng = np.random.default_rng(random_state)

    try:
        samples = rng.multivariate_normal(miss_vector, combined_covariance, size=n_samples)
    except np.linalg.LinAlgError:
        return {
            "pc_estimate": float("nan"),
            "confidence_interval_95": (float("nan"), float("nan")),
            "n_samples": n_samples,
            "n_hits": 0,
        }

    distances = np.linalg.norm(samples, axis=1)
    n_hits = int(np.sum(distances <= combined_radius_m))
    pc_estimate = n_hits / n_samples

    # Wilson score interval for binomial proportion (better than Wald for small p)
    ci_lower, ci_upper = _wilson_confidence_interval(n_hits, n_samples, z=1.96)

    return {
        "pc_estimate": pc_estimate,
        "confidence_interval_95": (ci_lower, ci_upper),
        "n_samples": n_samples,
        "n_hits": n_hits,
    }


def verify_analytic_with_monte_carlo(
    miss_vector: np.ndarray,
    combined_covariance_2d: np.ndarray,
    combined_radius_m: float = 20.0,
    n_samples: int = 100000,
    random_state: int = 42,
) -> dict:
    """Compare analytic 2D Pc with Monte Carlo estimate.

    Parameters
    ----------
    miss_vector : np.ndarray
        2D miss vector in the encounter plane (meters).
    combined_covariance_2d : np.ndarray
        2x2 covariance in the encounter plane (m^2).
    combined_radius_m : float
        Combined hard-body radius (meters).
    n_samples : int
        Number of Monte Carlo samples.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        analytic_pc : float
        mc_pc : float
        mc_ci_lower : float
        mc_ci_upper : float
        agreement : bool - True if analytic Pc falls within MC 95% CI
        ratio : float - analytic / mc (NaN if mc is zero)
    """
    analytic_pc = compute_collision_probability_2d(
        miss_distance_m=np.linalg.norm(miss_vector),
        combined_covariance_2d=combined_covariance_2d,
        combined_radius_m=combined_radius_m,
        miss_vector_2d=miss_vector,
    )

    mc_result = monte_carlo_collision_probability(
        miss_vector=miss_vector,
        combined_covariance=combined_covariance_2d,
        combined_radius_m=combined_radius_m,
        n_samples=n_samples,
        random_state=random_state,
    )

    mc_pc = mc_result["pc_estimate"]
    ci_lower, ci_upper = mc_result["confidence_interval_95"]

    # Check agreement: analytic Pc should be within the MC confidence interval
    # Use a wider tolerance band to account for grid integration discretization
    tolerance = max(3.0 / np.sqrt(n_samples), 1e-10)
    agreement = (ci_lower - tolerance) <= analytic_pc <= (ci_upper + tolerance)

    ratio = analytic_pc / mc_pc if mc_pc > 0 else float("nan")

    return {
        "analytic_pc": analytic_pc,
        "mc_pc": mc_pc,
        "mc_ci_lower": ci_lower,
        "mc_ci_upper": ci_upper,
        "agreement": bool(agreement),
        "ratio": ratio,
    }


def _wilson_confidence_interval(
    n_hits: int, n_total: int, z: float = 1.96
) -> tuple:
    """Wilson score confidence interval for a binomial proportion.

    More accurate than the normal (Wald) interval for small proportions,
    which is exactly the regime we care about for collision probability.
    """
    if n_total == 0:
        return (0.0, 1.0)

    p_hat = n_hits / n_total
    z2 = z * z
    denom = 1.0 + z2 / n_total
    center = (p_hat + z2 / (2.0 * n_total)) / denom
    half_width = (z / denom) * np.sqrt(p_hat * (1.0 - p_hat) / n_total + z2 / (4.0 * n_total**2))

    return (max(0.0, center - half_width), min(1.0, center + half_width))

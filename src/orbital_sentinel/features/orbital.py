"""Orbital-element-derived features."""

import numpy as np
import pandas as pd

R_EARTH_KM = 6378.137
MU_EARTH = 398600.4418  # km^3/s^2


def engineer_orbital_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features from orbital elements."""
    df = df.copy()

    for prefix in ("t", "c"):
        sma_col = f"{prefix}_j2k_sma"
        ecc_col = f"{prefix}_j2k_ecc"

        if sma_col in df.columns:
            sma = df[sma_col]
            positive_sma = sma.clip(lower=1.0)
            df[f"{prefix}_orbital_period"] = 2 * np.pi * np.sqrt(positive_sma ** 3 / MU_EARTH)
            df[f"{prefix}_mean_motion"] = np.sqrt(MU_EARTH / positive_sma ** 3)

        if sma_col in df.columns and ecc_col in df.columns:
            sma = df[sma_col]
            ecc = df[ecc_col].clip(lower=0.0, upper=0.999)
            df[f"{prefix}_perigee_alt"] = sma * (1 - ecc) - R_EARTH_KM
            df[f"{prefix}_apogee_alt"] = sma * (1 + ecc) - R_EARTH_KM
            df[f"{prefix}_orbit_energy"] = -MU_EARTH / (2 * sma.clip(lower=1.0))

    if "t_j2k_sma" in df.columns and "c_j2k_sma" in df.columns:
        df["sma_difference"] = (df["t_j2k_sma"] - df["c_j2k_sma"]).abs()

    if "t_j2k_inc" in df.columns and "c_j2k_inc" in df.columns:
        df["inclination_difference"] = (df["t_j2k_inc"] - df["c_j2k_inc"]).abs()

    if "t_h_apo" in df.columns and "t_h_per" in df.columns:
        h_apo = df["t_h_apo"].clip(lower=1.0)
        h_per = df["t_h_per"].clip(lower=1.0)
        df["t_orbit_eccentricity_proxy"] = (h_apo - h_per) / (h_apo + h_per)

    if "c_h_apo" in df.columns and "c_h_per" in df.columns:
        h_apo = df["c_h_apo"].clip(lower=1.0)
        h_per = df["c_h_per"].clip(lower=1.0)
        df["c_orbit_eccentricity_proxy"] = (h_apo - h_per) / (h_apo + h_per)

    return df

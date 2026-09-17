"""Covariance-derived features for conjunction risk assessment."""

import numpy as np
import pandas as pd


def engineer_covariance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features from covariance and uncertainty information."""
    df = df.copy()

    for prefix in ("t", "c"):
        sr = f"{prefix}_sigma_r"
        st = f"{prefix}_sigma_t"
        sn = f"{prefix}_sigma_n"

        if all(c in df.columns for c in [sr, st, sn]):
            df[f"{prefix}_position_uncertainty"] = np.sqrt(
                df[sr] ** 2 + df[st] ** 2 + df[sn] ** 2
            )
            df[f"{prefix}_sigma_max"] = df[[sr, st, sn]].max(axis=1)
            df[f"{prefix}_sigma_min"] = df[[sr, st, sn]].min(axis=1)
            sigma_min = df[f"{prefix}_sigma_min"].replace(0, np.nan)
            df[f"{prefix}_sigma_ratio"] = df[f"{prefix}_sigma_max"] / sigma_min

    if "t_position_uncertainty" in df.columns and "c_position_uncertainty" in df.columns:
        df["combined_position_uncertainty"] = np.sqrt(
            df["t_position_uncertainty"] ** 2 + df["c_position_uncertainty"] ** 2
        )

    if "t_sigma_r" in df.columns and "c_sigma_r" in df.columns:
        df["combined_sigma_r"] = np.sqrt(df["t_sigma_r"] ** 2 + df["c_sigma_r"] ** 2)

    if "t_sigma_t" in df.columns and "c_sigma_t" in df.columns:
        df["combined_sigma_t"] = np.sqrt(df["t_sigma_t"] ** 2 + df["c_sigma_t"] ** 2)

    if "t_sigma_n" in df.columns and "c_sigma_n" in df.columns:
        df["combined_sigma_n"] = np.sqrt(df["t_sigma_n"] ** 2 + df["c_sigma_n"] ** 2)

    if "miss_distance" in df.columns and "combined_position_uncertainty" in df.columns:
        safe_unc = df["combined_position_uncertainty"].replace(0, np.nan)
        df["miss_distance_sigma_ratio"] = df["miss_distance"] / safe_unc

    if "t_position_covariance_det" in df.columns:
        det = df["t_position_covariance_det"].clip(lower=1e-100)
        df["t_log_cov_det"] = np.log10(det)

    if "c_position_covariance_det" in df.columns:
        det = df["c_position_covariance_det"].clip(lower=1e-100)
        df["c_log_cov_det"] = np.log10(det)

    return df

"""Static (non-temporal) derived features from conjunction geometry and object properties."""

import numpy as np
import pandas as pd


def engineer_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from raw conjunction data columns."""
    df = df.copy()

    if "relative_position_r" in df.columns and "relative_position_t" in df.columns and "relative_position_n" in df.columns:
        df["relative_position_magnitude"] = np.sqrt(
            df["relative_position_r"] ** 2
            + df["relative_position_t"] ** 2
            + df["relative_position_n"] ** 2
        )

    if "relative_velocity_r" in df.columns and "relative_velocity_t" in df.columns and "relative_velocity_n" in df.columns:
        df["relative_velocity_magnitude"] = np.sqrt(
            df["relative_velocity_r"] ** 2
            + df["relative_velocity_t"] ** 2
            + df["relative_velocity_n"] ** 2
        )

    if "miss_distance" in df.columns and "relative_speed" in df.columns:
        safe_speed = df["relative_speed"].replace(0, np.nan)
        df["encounter_duration_proxy"] = df["miss_distance"] / safe_speed

    if "t_obs_available" in df.columns and "t_obs_used" in df.columns:
        safe_avail = df["t_obs_available"].replace(0, np.nan)
        df["t_obs_utilization"] = df["t_obs_used"] / safe_avail

    if "c_obs_available" in df.columns and "c_obs_used" in df.columns:
        safe_avail = df["c_obs_available"].replace(0, np.nan)
        df["c_obs_utilization"] = df["c_obs_used"] / safe_avail

    if "t_recommended_od_span" in df.columns and "t_actual_od_span" in df.columns:
        safe_rec = df["t_recommended_od_span"].replace(0, np.nan)
        df["t_od_span_ratio"] = df["t_actual_od_span"] / safe_rec

    if "c_recommended_od_span" in df.columns and "c_actual_od_span" in df.columns:
        safe_rec = df["c_recommended_od_span"].replace(0, np.nan)
        df["c_od_span_ratio"] = df["c_actual_od_span"] / safe_rec

    if "mahalanobis_distance" in df.columns and "miss_distance" in df.columns:
        safe_mahal = df["mahalanobis_distance"].replace(0, np.nan)
        df["miss_to_mahalanobis_ratio"] = df["miss_distance"] / safe_mahal

    return df

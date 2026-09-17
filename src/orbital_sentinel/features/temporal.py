"""Temporal features derived from time-to-TCA and event sequences."""

import numpy as np
import pandas as pd


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive temporal features from time_to_tca and event-level grouping."""
    df = df.copy()

    if "time_to_tca" in df.columns:
        tca = df["time_to_tca"]
        df["time_to_tca_hours"] = tca * 24.0
        df["time_to_tca_squared"] = tca ** 2
        df["log_time_to_tca"] = np.log1p(tca.clip(lower=0))
        df["is_close_approach"] = (tca <= 0.5).astype(np.float64)

    if "event_id" in df.columns and "time_to_tca" in df.columns:
        df = df.sort_values(["event_id", "time_to_tca"], ascending=[True, False])
        df["cdm_sequence_number"] = df.groupby("event_id").cumcount()
        df["cdm_total_in_event"] = df.groupby("event_id")["event_id"].transform("count")
        df["cdm_sequence_fraction"] = df["cdm_sequence_number"] / df["cdm_total_in_event"].clip(lower=1)

    if "t_time_lastob_start" in df.columns and "t_time_lastob_end" in df.columns:
        df["t_observation_span"] = df["t_time_lastob_end"] - df["t_time_lastob_start"]

    if "c_time_lastob_start" in df.columns and "c_time_lastob_end" in df.columns:
        df["c_observation_span"] = df["c_time_lastob_end"] - df["c_time_lastob_start"]

    return df

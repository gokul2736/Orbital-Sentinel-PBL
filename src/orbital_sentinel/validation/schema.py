import pandas as pd

REQUIRED_COLUMNS = [
    "event_id",
    "time_to_tca",
    "risk",
    "miss_distance",
    "relative_speed",
    "relative_position_r",
    "relative_position_t",
    "relative_position_n",
    "relative_velocity_r",
    "relative_velocity_t",
    "relative_velocity_n",
    "t_j2k_sma",
    "t_j2k_ecc",
    "t_j2k_inc",
    "c_object_type",
    "mahalanobis_distance",
]


def validate_schema(df: pd.DataFrame) -> list:
    issues = []

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")

    if "risk" in df.columns and not pd.api.types.is_numeric_dtype(df["risk"]):
        issues.append("Column 'risk' is not numeric")

    if "event_id" in df.columns and df["event_id"].isnull().any():
        null_count = df["event_id"].isnull().sum()
        issues.append(f"Column 'event_id' has {null_count} null values")

    return issues

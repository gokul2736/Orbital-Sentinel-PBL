from orbital_sentinel.features.static import engineer_static_features
from orbital_sentinel.features.orbital import engineer_orbital_features
from orbital_sentinel.features.covariance import engineer_covariance_features
from orbital_sentinel.features.temporal import engineer_temporal_features


def engineer_all_features(df):
    """Apply all feature engineering steps to a DataFrame."""
    df = engineer_static_features(df)
    df = engineer_orbital_features(df)
    df = engineer_covariance_features(df)
    df = engineer_temporal_features(df)
    return df

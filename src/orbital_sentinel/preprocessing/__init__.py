import numpy as np

from orbital_sentinel.config.settings import get_settings
from orbital_sentinel.ingestion.dataset_loader import load_esa_kelvins
from orbital_sentinel.features import engineer_all_features
from orbital_sentinel.preprocessing.cleaning import clean_dataframe
from orbital_sentinel.preprocessing.normalization import get_feature_columns, fit_scaler, normalize
from orbital_sentinel.preprocessing.splitting import event_based_split


def get_preprocessed_data() -> dict:
    settings = get_settings()

    df = load_esa_kelvins(split="train")
    df = clean_dataframe(df)
    df = engineer_all_features(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    feature_cols = get_feature_columns(df)

    train_df, val_df = event_based_split(
        df,
        test_size=settings.splitting.test_size,
        random_state=settings.splitting.random_state,
    )

    target_col = settings.target.column

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values

    scaler = fit_scaler(X_train)
    X_train = normalize(X_train, scaler)
    X_val = normalize(X_val, scaler)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "feature_names": feature_cols,
        "scaler": scaler,
        "train_df": train_df,
        "val_df": val_df,
    }

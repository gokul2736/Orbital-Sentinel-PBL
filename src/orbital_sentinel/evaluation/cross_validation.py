"""K-fold cross-validation with event-based splitting."""

from typing import Optional

import numpy as np
import pandas as pd


def event_kfold_split(
    df: pd.DataFrame,
    n_folds: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate K-fold splits based on event IDs to prevent data leakage."""
    events = df["event_id"].unique()
    rng = np.random.RandomState(random_state)
    rng.shuffle(events)

    fold_size = len(events) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(events)
        val_events = set(events[start:end])
        val_mask = df["event_id"].isin(val_events)
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        folds.append((train_idx, val_idx))

    return folds


def cross_validate_model(
    model_class,
    model_kwargs: dict,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    fit_kwargs: Optional[dict] = None,
) -> dict:
    """Run K-fold cross-validation for a model.

    Returns per-fold and aggregated metrics.
    """
    from orbital_sentinel.evaluation.metrics import evaluate_regression

    fold_metrics = []
    predictions = np.full(len(y), np.nan)

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_class(**model_kwargs)

        fkw = dict(fit_kwargs or {})
        if hasattr(model, "early_stopping_rounds") and model.early_stopping_rounds:
            fkw["eval_set"] = [(X_val, y_val)]

        model.fit(X_train, y_train, **fkw)
        pred = model.predict(X_val)
        predictions[val_idx] = pred

        metrics = evaluate_regression(y_val, pred)
        metrics["fold"] = fold_idx
        metrics["n_train"] = len(train_idx)
        metrics["n_val"] = len(val_idx)
        fold_metrics.append(metrics)

    valid_mask = ~np.isnan(predictions)
    overall = evaluate_regression(y[valid_mask], predictions[valid_mask])

    rmse_values = [m["rmse"] for m in fold_metrics]
    mae_values = [m["mae"] for m in fold_metrics]
    r2_values = [m["r2"] for m in fold_metrics]

    return {
        "fold_metrics": fold_metrics,
        "overall": overall,
        "n_folds": len(folds),
        "mean_rmse": float(np.mean(rmse_values)),
        "std_rmse": float(np.std(rmse_values)),
        "mean_mae": float(np.mean(mae_values)),
        "std_mae": float(np.std(mae_values)),
        "mean_r2": float(np.mean(r2_values)),
        "std_r2": float(np.std(r2_values)),
        "oof_predictions": predictions,
    }


def compare_models_cv(
    model_configs: dict,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Cross-validate multiple models and compare results.

    model_configs: {name: {"class": ModelClass, "kwargs": {...}, "fit_kwargs": {...}}}
    """
    results = {}
    for name, config in model_configs.items():
        cv_result = cross_validate_model(
            model_class=config["class"],
            model_kwargs=config.get("kwargs", {}),
            X=X,
            y=y,
            folds=folds,
            fit_kwargs=config.get("fit_kwargs"),
        )
        results[name] = {
            "mean_rmse": cv_result["mean_rmse"],
            "std_rmse": cv_result["std_rmse"],
            "mean_mae": cv_result["mean_mae"],
            "std_mae": cv_result["std_mae"],
            "mean_r2": cv_result["mean_r2"],
            "std_r2": cv_result["std_r2"],
            "fold_metrics": cv_result["fold_metrics"],
        }

    return results

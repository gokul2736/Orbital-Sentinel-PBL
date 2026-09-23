"""Hyperparameter tuning with Optuna for all model types."""

from typing import Optional

import numpy as np
from sklearn.model_selection import cross_val_score


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    random_state: int = 42,
) -> dict:
    """Tune XGBoost hyperparameters using Optuna."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

        from orbital_sentinel.models.baselines.xgboost_model import XGBoostModel
        model = XGBoostModel(**params, random_state=random_state, early_stopping_rounds=30)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        pred = model.predict(X_val)
        rmse = float(np.sqrt(np.mean((y_val - pred) ** 2)))
        return rmse

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)

    return {
        "best_params": study.best_params,
        "best_rmse": study.best_value,
        "n_trials": n_trials,
        "study": study,
    }


def tune_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    random_state: int = 42,
) -> dict:
    """Tune LightGBM hyperparameters using Optuna."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 20, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

        from orbital_sentinel.models.baselines.lightgbm_model import LightGBMModel
        model = LightGBMModel(**params, random_state=random_state, early_stopping_rounds=30)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        pred = model.predict(X_val)
        rmse = float(np.sqrt(np.mean((y_val - pred) ** 2)))
        return rmse

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)

    return {
        "best_params": study.best_params,
        "best_rmse": study.best_value,
        "n_trials": n_trials,
        "study": study,
    }


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 30,
    random_state: int = 42,
) -> dict:
    """Tune Random Forest hyperparameters using Optuna."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 6, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }

        from orbital_sentinel.models.baselines.random_forest import RandomForestModel
        model = RandomForestModel(**params, random_state=random_state)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        rmse = float(np.sqrt(np.mean((y_val - pred) ** 2)))
        return rmse

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)

    return {
        "best_params": study.best_params,
        "best_rmse": study.best_value,
        "n_trials": n_trials,
        "study": study,
    }

"""SHAP-based model explainability for collision risk predictions."""

from typing import List

import numpy as np


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: List[str],
    max_samples: int = 500,
) -> dict:
    import shap

    if X.shape[0] > max_samples:
        indices = np.random.default_rng(42).choice(X.shape[0], max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.KernelExplainer(model.predict, X_sample[:min(100, len(X_sample))])

    sv = explainer.shap_values(X_sample)

    if isinstance(sv, list):
        sv = sv[0] if len(sv) == 1 else sv[-1]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(expected_value[0]) if len(expected_value) == 1 else float(expected_value[-1])
    else:
        expected_value = float(expected_value)

    return {
        "shap_values": sv,
        "feature_names": list(feature_names),
        "expected_value": expected_value,
    }


def top_features_for_prediction(
    shap_values: np.ndarray,
    feature_names: List[str],
    idx: int,
    top_k: int = 10,
) -> List[dict]:
    sv_row = shap_values[idx]
    order = np.argsort(np.abs(sv_row))[::-1][:top_k]

    results = []
    for i in order:
        results.append({
            "feature": feature_names[i],
            "shap_value": float(sv_row[i]),
            "direction": "increases_risk" if sv_row[i] > 0 else "decreases_risk",
        })
    return results


def global_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_k: int = 20,
) -> List[dict]:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]

    results = []
    for rank, i in enumerate(order, 1):
        results.append({
            "feature": feature_names[i],
            "mean_abs_shap": float(mean_abs[i]),
            "rank": rank,
        })
    return results

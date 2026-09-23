"""Feature selection using SHAP importance and recursive elimination."""

from typing import Optional

import numpy as np


def shap_feature_selection(
    model,
    X_train: np.ndarray,
    feature_names: list[str],
    top_k: int = 50,
    max_samples: int = 500,
) -> dict:
    """Select top-k features by mean absolute SHAP value."""
    from orbital_sentinel.explainability.shap_analysis import compute_shap_values, global_feature_importance

    sample_idx = np.random.RandomState(42).choice(len(X_train), size=min(max_samples, len(X_train)), replace=False)
    X_sample = X_train[sample_idx]

    shap_result = compute_shap_values(model, X_sample, feature_names, max_samples=max_samples)
    importance = global_feature_importance(shap_result["shap_values"], feature_names, top_k=len(feature_names))

    ranked = sorted(importance, key=lambda x: x["mean_abs_shap"], reverse=True)
    selected = [f["feature"] for f in ranked[:top_k]]
    selected_indices = [feature_names.index(f) for f in selected]

    return {
        "selected_features": selected,
        "selected_indices": selected_indices,
        "n_original": len(feature_names),
        "n_selected": len(selected),
        "importance_ranking": ranked,
    }


def variance_threshold_selection(
    X: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.01,
) -> dict:
    """Remove features with variance below threshold."""
    variances = np.var(X, axis=0)
    mask = variances > threshold
    selected = [f for f, keep in zip(feature_names, mask) if keep]
    selected_indices = [i for i, keep in enumerate(mask) if keep]

    return {
        "selected_features": selected,
        "selected_indices": selected_indices,
        "n_original": len(feature_names),
        "n_selected": len(selected),
        "removed": [f for f, keep in zip(feature_names, mask) if not keep],
    }


def correlation_filter(
    X: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.95,
) -> dict:
    """Remove one of each pair of features with correlation above threshold."""
    corr = np.corrcoef(X.T)
    n = len(feature_names)
    to_remove = set()

    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            if abs(corr[i, j]) > threshold:
                to_remove.add(j)

    selected_indices = [i for i in range(n) if i not in to_remove]
    selected = [feature_names[i] for i in selected_indices]

    return {
        "selected_features": selected,
        "selected_indices": selected_indices,
        "n_original": n,
        "n_selected": len(selected),
        "removed": [feature_names[i] for i in to_remove],
        "n_correlated_pairs": len(to_remove),
    }


def apply_selection(X: np.ndarray, indices: list[int]) -> np.ndarray:
    """Apply feature selection by indices."""
    return X[:, indices]

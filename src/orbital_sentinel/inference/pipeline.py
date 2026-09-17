"""Single-event and batch inference pipeline for live predictions."""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


def load_inference_artifacts(
    models_dir: str = "models_saved",
    model_name: str = "XGBoost",
) -> dict:
    """Load trained model, scaler, and feature names for inference."""
    models_path = Path(models_dir)

    model_path = models_path / f"{model_name}_model.joblib"
    scaler_path = models_path / "scaler.joblib"
    features_path = models_path / "feature_names.json"

    artifacts = {"model": None, "scaler": None, "feature_names": None, "model_name": model_name}

    if model_path.exists():
        artifacts["model"] = joblib.load(str(model_path))
    if scaler_path.exists():
        artifacts["scaler"] = joblib.load(str(scaler_path))
    if features_path.exists():
        with open(features_path) as f:
            artifacts["feature_names"] = json.load(f)

    return artifacts


def predict_single_event(
    row_dict: dict,
    model,
    scaler,
    feature_names: list,
    include_physics: bool = True,
    include_explanation: bool = True,
) -> dict:
    """Run full inference on a single conjunction event.

    Returns dict with prediction, category, interval, physics, fusion, and explanation.
    """
    from orbital_sentinel.preprocessing.cleaning import clean_dataframe
    from orbital_sentinel.features import engineer_all_features
    from orbital_sentinel.evaluation.thresholding import apply_risk_categories

    single_df = pd.DataFrame([row_dict])
    single_clean = clean_dataframe(single_df)
    single_feat = engineer_all_features(single_clean)

    for col in feature_names:
        if col not in single_feat.columns:
            single_feat[col] = 0.0
    X = single_feat[feature_names].values

    if scaler is not None:
        X = scaler.transform(X)

    prediction = float(model.predict(X)[0])
    categories = apply_risk_categories(np.array([prediction]))
    risk_category = categories[0]

    result = {
        "prediction": prediction,
        "risk_category": risk_category,
        "confidence": None,
        "interval": None,
        "physics_result": None,
        "fusion": None,
        "explanation": None,
        "top_features": None,
    }

    try:
        from orbital_sentinel.uncertainty.predictive import (
            estimate_prediction_interval,
            prediction_confidence,
        )
        lower, upper = estimate_prediction_interval(model, X, confidence=0.9)
        result["interval"] = (float(lower[0]), float(upper[0]))
        conf = prediction_confidence(model, X)
        result["confidence"] = float(conf[0])
    except Exception:
        result["interval"] = (prediction - 2.0, prediction + 2.0)
        result["confidence"] = 0.5

    if include_physics:
        try:
            from orbital_sentinel.physics import verify_conjunction_physics
            result["physics_result"] = verify_conjunction_physics(row_dict)
        except Exception:
            pass

    try:
        from orbital_sentinel.fusion import fuse_risk
        ml_interval = result["interval"] or (prediction - 2.0, prediction + 2.0)
        fusion = fuse_risk(prediction, ml_interval, physics_result=result["physics_result"])
        if hasattr(fusion, "__dict__"):
            result["fusion"] = fusion.__dict__
        elif isinstance(fusion, dict):
            result["fusion"] = fusion
    except Exception:
        pass

    if include_explanation:
        try:
            from orbital_sentinel.explainability.shap_analysis import (
                compute_shap_values,
                top_features_for_prediction,
            )
            from orbital_sentinel.explainability import explain_prediction, generate_decision_summary

            shap_result = compute_shap_values(model, X, feature_names, max_samples=100)
            top_feats = top_features_for_prediction(
                shap_result["shap_values"], feature_names, idx=0, top_k=5,
            )
            result["top_features"] = top_feats

            assessment = result["fusion"] or {
                "risk_category": risk_category,
                "fused_risk": prediction,
                "confidence": result["confidence"],
            }
            result["explanation"] = explain_prediction(
                assessment, top_feats, result["physics_result"],
            )
        except Exception:
            pass

    return result


def predict_batch(
    df: pd.DataFrame,
    model,
    scaler,
    feature_names: list,
) -> list:
    """Run inference on a batch of conjunction events. Returns list of result dicts."""
    from orbital_sentinel.preprocessing.cleaning import clean_dataframe
    from orbital_sentinel.features import engineer_all_features
    from orbital_sentinel.evaluation.thresholding import apply_risk_categories

    cleaned = clean_dataframe(df)
    featured = engineer_all_features(cleaned)

    for col in feature_names:
        if col not in featured.columns:
            featured[col] = 0.0
    X = featured[feature_names].values

    if scaler is not None:
        X = scaler.transform(X)

    predictions = model.predict(X)
    categories = apply_risk_categories(predictions)

    results = []
    for i in range(len(predictions)):
        results.append({
            "index": i,
            "prediction": float(predictions[i]),
            "risk_category": categories[i],
        })

    return results

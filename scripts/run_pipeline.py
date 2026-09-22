"""
Orbital Sentinel — End-to-End Pipeline Runner

Executes the full conjunction risk assessment pipeline:
  1. Load ESA Kelvins dataset
  2. Validate schema and quality
  3. Clean and preprocess
  4. Engineer features
  5. Split by event (no leakage)
  6. Train baseline models
  7. Evaluate models
  8. Run physics verification on sample
  9. Fuse risk assessments
  10. Generate explanations
  11. Save results and model artifacts
"""

import sys
import os
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import joblib


def main():
    start_time = time.time()
    project_root = Path(__file__).resolve().parent.parent
    results = {}

    print("=" * 70)
    print("ORBITAL SENTINEL — End-to-End Pipeline")
    print("=" * 70)

    # ── Step 1: Load Data ──
    print("\n[1/11] Loading ESA Kelvins dataset...")
    from orbital_sentinel.ingestion.dataset_loader import load_esa_kelvins
    train_df = load_esa_kelvins(split="train")
    test_df = load_esa_kelvins(split="test")
    print(f"  Train: {train_df.shape[0]:,} rows, {train_df.shape[1]} columns")
    print(f"  Test:  {test_df.shape[0]:,} rows, {test_df.shape[1]} columns")

    # ── Step 2: Validate ──
    print("\n[2/11] Validating schema and quality...")
    from orbital_sentinel.validation.schema import validate_schema
    from orbital_sentinel.validation.quality import assess_quality
    schema_issues = validate_schema(train_df)
    if schema_issues:
        print(f"  Schema issues: {schema_issues}")
    else:
        print("  Schema validation: PASSED")
    quality = assess_quality(train_df)
    print(f"  Null columns: {quality.get('columns_with_nulls', 'N/A')}")
    print(f"  Risk floor pct: {quality.get('risk_floor_pct', 'N/A'):.1f}%")
    results["validation"] = {"schema_issues": schema_issues, "quality": quality}

    # ── Step 3: Check leakage ──
    print("\n[3/11] Checking for target leakage...")
    from orbital_sentinel.validation.leakage import check_leakage
    from orbital_sentinel.preprocessing.normalization import get_feature_columns
    from orbital_sentinel.preprocessing.cleaning import clean_dataframe
    cleaned = clean_dataframe(train_df)
    feature_cols = get_feature_columns(cleaned)
    leakage_warnings = check_leakage(cleaned, feature_cols)
    if leakage_warnings:
        print(f"  Leakage warnings: {leakage_warnings}")
    else:
        print("  No leakage detected in feature set")

    # ── Step 4: Preprocess ──
    print("\n[4/11] Preprocessing and feature engineering...")
    from orbital_sentinel.preprocessing import get_preprocessed_data
    data = get_preprocessed_data()
    X_train = data["X_train"]
    X_val = data["X_val"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    feature_names = data["feature_names"]
    scaler = data["scaler"]
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  Features: {len(feature_names)}")

    # ── Step 5: Train Models ──
    print("\n[5/11] Training baseline models...")
    from orbital_sentinel.models import train_all_baselines
    model_results = train_all_baselines(X_train, y_train, X_val, y_val, feature_names)

    best_model_name = None
    best_rmse = float("inf")
    for name, (model, train_metrics, val_metrics) in model_results.items():
        rmse = val_metrics.get("rmse", float("inf"))
        print(f"\n  {name}:")
        print(f"    Train — MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
        print(f"    Val   — MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name

    print(f"\n  Best model: {best_model_name} (val RMSE: {best_rmse:.4f})")
    best_model = model_results[best_model_name][0]
    results["model_metrics"] = {
        name: {"train": tm, "val": vm}
        for name, (_, tm, vm) in model_results.items()
    }

    # ── Step 6: Evaluate best model ──
    print("\n[6/11] Detailed evaluation of best model...")
    from orbital_sentinel.evaluation.metrics import evaluate_regression, evaluate_by_risk_band, evaluate_at_threshold
    val_pred = best_model.predict(X_val)
    full_metrics = evaluate_regression(y_val, val_pred)
    band_metrics = evaluate_by_risk_band(y_val, val_pred)
    threshold_metrics = evaluate_at_threshold(y_val, val_pred, threshold=-5.0)
    print(f"  Overall — MAE: {full_metrics['mae']:.4f}, RMSE: {full_metrics['rmse']:.4f}, R²: {full_metrics['r2']:.4f}")
    print(f"  Correlation: {full_metrics.get('correlation', 'N/A'):.4f}")
    for band_name, bm in band_metrics.items():
        print(f"  Band '{band_name}' — MAE: {bm.get('mae', 'N/A'):.4f}, count: {bm.get('count', 0)}")
    print(f"  Binary @-5.0 — Precision: {threshold_metrics.get('precision', 'N/A'):.4f}, "
          f"Recall: {threshold_metrics.get('recall', 'N/A'):.4f}, "
          f"F1: {threshold_metrics.get('f1', 'N/A'):.4f}")
    results["evaluation"] = {
        "overall": full_metrics,
        "by_band": band_metrics,
        "threshold_-5": threshold_metrics,
    }

    # ── Step 7: Uncertainty estimation ──
    print("\n[7/11] Estimating prediction uncertainty...")
    from orbital_sentinel.uncertainty.predictive import estimate_prediction_interval
    lower, upper = estimate_prediction_interval(best_model, X_val[:100])
    interval_widths = upper - lower
    print(f"  Mean interval width (sample): {np.mean(interval_widths):.4f}")
    print(f"  Median interval width: {np.median(interval_widths):.4f}")

    # ── Step 8: Physics verification ──
    print("\n[8/11] Running physics verification on sample conjunctions...")
    from orbital_sentinel.physics import verify_conjunction_physics
    sample_indices = np.random.RandomState(42).choice(len(X_val), size=min(20, len(X_val)), replace=False)

    physics_results = []
    val_df_for_physics = data.get("val_df")
    if val_df_for_physics is not None and len(val_df_for_physics) > 0:
        for idx in sample_indices:
            if idx < len(val_df_for_physics):
                row = val_df_for_physics.iloc[idx].to_dict()
                try:
                    pr = verify_conjunction_physics(row)
                    physics_results.append(pr)
                except Exception as e:
                    physics_results.append({"error": str(e)})
        valid_physics = [p for p in physics_results if "error" not in p and p.get("analytic_pc") is not None]
        print(f"  Physics verified: {len(valid_physics)}/{len(physics_results)} conjunctions")
        if valid_physics:
            pcs = [p["log10_pc"] for p in valid_physics if p.get("log10_pc") is not None]
            if pcs:
                print(f"  Analytic log10(Pc) range: [{min(pcs):.2f}, {max(pcs):.2f}]")
    else:
        print("  Physics verification skipped (val_df not available in preprocessed data)")
    results["physics_sample_count"] = len(physics_results)

    # ── Step 9: Risk fusion ──
    print("\n[9/11] Fusing risk assessments...")
    from orbital_sentinel.fusion import fuse_risk
    fusion_results = []
    for i in range(min(10, len(val_pred))):
        pred = float(val_pred[i])
        ml_interval = (pred - 2.0, pred + 2.0)
        phys = physics_results[i] if i < len(physics_results) and "error" not in physics_results[i] else None
        assessment = fuse_risk(pred, ml_interval, physics_result=phys)
        fusion_results.append(assessment)
    categories = {}
    for a in fusion_results:
        cat = a.risk_category if hasattr(a, "risk_category") else a.get("risk_category", "UNKNOWN")
        categories[cat] = categories.get(cat, 0) + 1
    print(f"  Sample risk categories: {categories}")

    # ── Step 10: Explainability ──
    print("\n[10/11] Generating explanations...")
    from orbital_sentinel.explainability import compute_shap_values, explain_prediction, generate_decision_summary
    try:
        shap_result = compute_shap_values(best_model, X_val[:200], feature_names, max_samples=200)
        print(f"  SHAP values computed for {shap_result['shap_values'].shape[0]} samples")

        from orbital_sentinel.explainability.shap_analysis import top_features_for_prediction, global_feature_importance
        top_global = global_feature_importance(shap_result["shap_values"], feature_names, top_k=10)
        print("  Top 10 global feature importances:")
        for feat in top_global:
            print(f"    {feat['rank']:2d}. {feat['feature']:40s} mean|SHAP|={feat['mean_abs_shap']:.4f}")

        top_local = top_features_for_prediction(shap_result["shap_values"], feature_names, idx=0, top_k=5)
        assessment_dict = fusion_results[0].__dict__ if hasattr(fusion_results[0], "__dict__") else fusion_results[0]
        explanation = explain_prediction(assessment_dict, top_local, physics_result=physics_results[0] if physics_results else None)
        print(f"\n  Sample explanation:\n  {explanation[:300]}...")

        summary = generate_decision_summary(assessment_dict)
        print(f"\n  Decision summary headline: {summary.get('headline', 'N/A')}")
        print(f"  Recommended action: {summary.get('recommended_action', 'N/A')}")
        results["explainability"] = {
            "top_features": top_global,
            "sample_explanation": explanation,
        }
    except Exception as e:
        print(f"  Explainability error: {e}")
        results["explainability"] = {"error": str(e)}

    # ── Step 11: Save artifacts ──
    print("\n[11/11] Saving artifacts...")
    artifacts_dir = project_root / "models_saved"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / f"{best_model_name}_model.joblib"
    best_model.save(str(model_path))
    print(f"  Model saved: {model_path}")

    scaler_path = artifacts_dir / "scaler.joblib"
    joblib.dump(scaler, str(scaler_path))
    print(f"  Scaler saved: {scaler_path}")

    feature_path = artifacts_dir / "feature_names.json"
    with open(feature_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"  Feature names saved: {feature_path}")

    results_path = project_root / "proofs" / "pipeline_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"  Results saved: {results_path}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"Best model: {best_model_name} — val RMSE: {best_rmse:.4f}")
    print(f"{'=' * 70}")

    return results


if __name__ == "__main__":
    main()

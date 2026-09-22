"""Training and evaluation commands."""

import click


@click.group()
def train():
    """Train and evaluate ML models."""


@train.command("run")
@click.option("--model", type=click.Choice(["xgboost", "lightgbm", "rf", "logistic", "all"]), default="all")
@click.option("--output-dir", type=click.Path(), default="models_saved")
def train_run(model: str, output_dir: str) -> None:
    """Run the full training pipeline."""
    from orbital_sentinel.cli.formatting import print_header, print_metric, print_success, print_error

    print_header("ORBITAL SENTINEL — Training Pipeline")

    try:
        import time
        start = time.time()

        click.echo("  Loading dataset...")
        from orbital_sentinel.ingestion.dataset_loader import load_esa_kelvins
        train_df = load_esa_kelvins(split="train")
        print_metric("Train rows", f"{len(train_df):,}")

        click.echo("  Preprocessing and feature engineering...")
        from orbital_sentinel.preprocessing import get_preprocessed_data
        data = get_preprocessed_data()
        print_metric("Features", str(len(data["feature_names"])))
        print_metric("X_train shape", str(data["X_train"].shape))
        print_metric("X_val shape", str(data["X_val"].shape))

        click.echo("  Training models...")
        from orbital_sentinel.models import train_all_baselines
        model_results = train_all_baselines(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
            data["feature_names"],
        )

        click.echo()
        best_name, best_rmse = None, float("inf")
        for name, (mdl, train_m, val_m) in model_results.items():
            rmse = val_m.get("rmse", float("inf"))
            click.echo(f"  {click.style(name, bold=True):30s} "
                        f"val_RMSE={rmse:.4f}  val_R2={val_m.get('r2', 0):.4f}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name

        click.echo()
        print_metric("Best model", f"{best_name} (RMSE: {best_rmse:.4f})", color="green")

        click.echo("  Saving artifacts...")
        from pathlib import Path
        import joblib
        import json

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        best_model = model_results[best_name][0]
        best_model.save(str(out / f"{best_name}_model.joblib"))
        joblib.dump(data["scaler"], str(out / "scaler.joblib"))
        with open(out / "feature_names.json", "w") as f:
            json.dump(data["feature_names"], f, indent=2)

        elapsed = time.time() - start
        print_success(f"Pipeline complete in {elapsed:.1f}s")

    except Exception as e:
        print_error(str(e))
        raise click.Abort()


@train.command("evaluate")
@click.option("--model-name", default="XGBoost")
@click.option("--models-dir", type=click.Path(), default="models_saved")
def train_evaluate(model_name: str, models_dir: str) -> None:
    """Evaluate a trained model."""
    from orbital_sentinel.cli.formatting import print_header, print_metric, print_error

    print_header(f"Evaluating: {model_name}")

    try:
        from orbital_sentinel.inference.pipeline import load_inference_artifacts
        artifacts = load_inference_artifacts(models_dir=models_dir, model_name=model_name)
        if artifacts["model"] is None:
            print_error(f"Model '{model_name}' not found in {models_dir}")
            return

        from orbital_sentinel.preprocessing import get_preprocessed_data
        data = get_preprocessed_data()

        predictions = artifacts["model"].predict(data["X_val"])

        from orbital_sentinel.evaluation.metrics import evaluate_regression, evaluate_by_risk_band, evaluate_at_threshold
        metrics = evaluate_regression(data["y_val"], predictions)
        band_metrics = evaluate_by_risk_band(data["y_val"], predictions)
        threshold_metrics = evaluate_at_threshold(data["y_val"], predictions, threshold=-5.0)

        print_metric("MAE", f"{metrics['mae']:.4f}")
        print_metric("RMSE", f"{metrics['rmse']:.4f}")
        print_metric("R-squared", f"{metrics['r2']:.4f}")
        print_metric("Correlation", f"{metrics.get('correlation', 0):.4f}")

        click.echo()
        click.echo(click.style("  Per-Risk-Band:", bold=True))
        for band, bm in band_metrics.items():
            click.echo(f"    {band:15s} MAE={bm.get('mae', 0):.4f}  count={bm.get('count', 0):,}")

        click.echo()
        click.echo(click.style("  Binary @-5.0 Threshold:", bold=True))
        print_metric("Precision", f"{threshold_metrics.get('precision', 0):.4f}")
        print_metric("Recall", f"{threshold_metrics.get('recall', 0):.4f}")
        print_metric("F1 Score", f"{threshold_metrics.get('f1', 0):.4f}")

    except Exception as e:
        print_error(str(e))


@train.command("compare")
def train_compare() -> None:
    """Compare all trained models."""
    from orbital_sentinel.cli.formatting import print_header, print_table, print_warning
    import json
    from pathlib import Path

    print_header("Model Comparison")

    results_path = Path("proofs/pipeline_results.json")
    if not results_path.exists():
        print_warning("No pipeline results found. Run training first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    model_metrics = results.get("model_metrics", {})
    if not model_metrics:
        print_warning("No model metrics in results.")
        return

    headers = ["Model", "Val RMSE", "Val MAE", "Val R2", "Train RMSE", "Gap"]
    rows = []
    for name, data in sorted(model_metrics.items(), key=lambda x: x[1].get("val", {}).get("rmse", 999)):
        val = data.get("val", {})
        trn = data.get("train", {})
        gap = val.get("rmse", 0) - trn.get("rmse", 0)
        rows.append([
            name,
            f"{val.get('rmse', 0):.4f}",
            f"{val.get('mae', 0):.4f}",
            f"{val.get('r2', 0):.4f}",
            f"{trn.get('rmse', 0):.4f}",
            f"{gap:.4f}",
        ])

    print_table(headers, rows, col_widths=[18, 12, 12, 12, 12, 10])

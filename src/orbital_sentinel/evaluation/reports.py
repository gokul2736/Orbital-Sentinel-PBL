"""Human-readable model performance reports."""

from typing import Optional

import pandas as pd


def generate_model_report(
    model_name: str,
    metrics: dict,
    val_metrics_by_band: Optional[dict] = None,
) -> str:
    """Generate a text report summarizing model performance."""
    lines = [
        f"=== {model_name} Performance Report ===",
        "",
        "Overall Metrics:",
        f"  MAE:         {metrics['mae']:.4f}",
        f"  RMSE:        {metrics['rmse']:.4f}",
        f"  R²:          {metrics['r2']:.4f}",
        f"  Median AE:   {metrics['median_ae']:.4f}",
        f"  Max Error:   {metrics['max_error']:.4f}",
        f"  Correlation: {metrics['correlation']:.4f}",
    ]

    if val_metrics_by_band:
        lines += ["", "Metrics by Risk Band:"]
        for band, band_metrics in val_metrics_by_band.items():
            n = band_metrics.get("n", 0)
            if n == 0:
                lines.append(f"  {band:>8s}: no samples")
                continue
            lines.append(
                f"  {band:>8s}: n={n:>6d}  MAE={band_metrics['mae']:.4f}  "
                f"RMSE={band_metrics['rmse']:.4f}  R²={band_metrics['r2']:.4f}"
            )

    return "\n".join(lines)


def compare_models(model_results: dict) -> pd.DataFrame:
    """Compare multiple models. Input: {name: (model, train_metrics, val_metrics)}."""
    rows = []
    for name, (_, train_m, val_m) in model_results.items():
        rows.append(
            {
                "model": name,
                "train_mae": train_m["mae"],
                "train_rmse": train_m["rmse"],
                "train_r2": train_m["r2"],
                "val_mae": val_m["mae"],
                "val_rmse": val_m["rmse"],
                "val_r2": val_m["r2"],
                "val_correlation": val_m["correlation"],
            }
        )
    df = pd.DataFrame(rows).set_index("model")
    df = df.sort_values("val_mae")
    return df

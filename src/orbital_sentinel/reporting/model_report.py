"""Generate HTML model performance reports."""

from typing import Optional
from pathlib import Path

from orbital_sentinel.reporting.templates import (
    html_wrapper, metric_card_html, table_html, bar_chart_html,
)


def generate_model_report(
    pipeline_results: dict,
    output_path: Optional[str] = None,
) -> str:
    """Generate an HTML report from pipeline results.

    Args:
        pipeline_results: Content of proofs/pipeline_results.json
        output_path: Optional file path to save the HTML

    Returns:
        HTML string of the report
    """
    model_metrics = pipeline_results.get("model_metrics", {})
    evaluation = pipeline_results.get("evaluation", {})
    explainability = pipeline_results.get("explainability", {})
    validation = pipeline_results.get("validation", {})

    sections = []

    # Best model highlight
    if model_metrics:
        best_name = min(model_metrics, key=lambda n: model_metrics[n].get("val", {}).get("rmse", 999))
        best_val = model_metrics[best_name].get("val", {})

        sections.append(f"""<h2>Best Model</h2>
<div class="metric-row">
    {metric_card_html("Model", best_name, "green")}
    {metric_card_html("Val RMSE", f"{best_val.get('rmse', 0):.4f}", "cyan")}
    {metric_card_html("Val R²", f"{best_val.get('r2', 0):.4f}", "purple")}
    {metric_card_html("Val MAE", f"{best_val.get('mae', 0):.4f}")}
</div>""")

    # Model comparison table
    if model_metrics:
        rows = []
        for name in sorted(model_metrics, key=lambda n: model_metrics[n].get("val", {}).get("rmse", 999)):
            val = model_metrics[name].get("val", {})
            trn = model_metrics[name].get("train", {})
            gap = val.get("rmse", 0) - trn.get("rmse", 0)
            rows.append([
                name,
                f"{trn.get('rmse', 0):.4f}",
                f"{val.get('rmse', 0):.4f}",
                f"{trn.get('r2', 0):.4f}",
                f"{val.get('r2', 0):.4f}",
                f"{val.get('mae', 0):.4f}",
                f"{val.get('correlation', 0):.4f}",
                f"{gap:.4f}",
            ])
        sections.append(f"""<h2>Model Comparison</h2>
{table_html(["Model", "Train RMSE", "Val RMSE", "Train R²", "Val R²", "Val MAE", "Correlation", "Gap"], rows, highlight_col=2)}""")

    # Per-risk-band performance
    by_band = evaluation.get("by_band", {})
    if by_band:
        rows = []
        for band_name, bm in by_band.items():
            rows.append([
                band_name.upper(),
                f"{bm.get('count', bm.get('n', 0)):,}",
                f"{bm.get('mae', 0):.4f}",
                f"{bm.get('rmse', 0):.4f}",
                f"{bm.get('correlation', 0):.4f}",
            ])
        sections.append(f"""<h2>Per-Risk-Band Performance</h2>
{table_html(["Band", "Count", "MAE", "RMSE", "Correlation"], rows)}""")

    # Binary classification
    thresh = evaluation.get("threshold_-5", {})
    if thresh:
        sections.append(f"""<h2>Binary Classification @ -5.0 Threshold</h2>
<div class="metric-row">
    {metric_card_html("Precision", f"{thresh.get('precision', 0):.4f}", "cyan")}
    {metric_card_html("Recall", f"{thresh.get('recall', 0):.4f}", "amber")}
    {metric_card_html("F1 Score", f"{thresh.get('f1', 0):.4f}", "green")}
    {metric_card_html("AUC", f"{thresh.get('auc', 0):.4f}", "purple")}
</div>""")

    # Feature importance
    top_features = explainability.get("top_features", [])
    if top_features:
        labels = [f["feature"] for f in top_features]
        values = [f["mean_abs_shap"] for f in top_features]
        colors_gradient = []
        max_v = max(values) if values else 1
        for v in values:
            ratio = v / max_v
            if ratio > 0.7:
                colors_gradient.append("#FF3366")
            elif ratio > 0.4:
                colors_gradient.append("#7C5CFC")
            else:
                colors_gradient.append("#00D4FF")

        sections.append(f"""<h2>Top 10 Global Feature Importance</h2>
{bar_chart_html(labels, values, colors_gradient)}""")

    # Data quality
    quality = validation.get("quality", {})
    if quality:
        sections.append(f"""<hr class="divider">
<h2>Data Quality Summary</h2>
<div class="metric-row">
    {metric_card_html("Total Rows", f"{quality.get('total_rows', 0):,}", "cyan")}
    {metric_card_html("Risk Floor %", f"{quality.get('risk_floor_pct', 0):.1f}%", "amber")}
    {metric_card_html("Neg Miss Dist", str(quality.get('negative_miss_distance_count', 0)))}
    {metric_card_html("Outlier Columns", str(len(quality.get('outlier_columns', []))))}
</div>""")

    body = "\n".join(sections)
    html = html_wrapper("Model Performance Report", body)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")

    return html

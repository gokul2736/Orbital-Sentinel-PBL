"""Generate HTML conjunction assessment reports."""

from typing import Optional
from pathlib import Path
import math

from orbital_sentinel.reporting.templates import (
    html_wrapper, risk_badge_html, metric_card_html, table_html, bar_chart_html, alert_html,
)


def generate_conjunction_report(
    prediction_result: dict,
    event_data: dict,
    output_path: Optional[str] = None,
) -> str:
    """Generate an HTML report for a single conjunction event assessment.

    Args:
        prediction_result: Output from predict_single_event()
        event_data: Raw CDM row data
        output_path: Optional file path to save the HTML

    Returns:
        HTML string of the report
    """
    pred = prediction_result.get("prediction", 0)
    category = prediction_result.get("risk_category", "UNKNOWN")
    confidence = prediction_result.get("confidence")
    interval = prediction_result.get("interval")
    physics = prediction_result.get("physics_result")
    fusion = prediction_result.get("fusion")
    explanation = prediction_result.get("explanation")
    top_features = prediction_result.get("top_features")

    event_id = event_data.get("event_id", "N/A")
    color = "red" if pred > -5 else ("amber" if pred > -7 else ("cyan" if pred > -15 else "green"))

    sections = []

    # Executive summary
    rec = "IMMEDIATE REVIEW REQUIRED" if pred > -5 else ("MONITOR CLOSELY" if pred > -7 else "ROUTINE MONITORING")
    sections.append(f"""<h2>Executive Summary</h2>
<div class="metric-row">
    {metric_card_html("Event ID", str(event_id))}
    {metric_card_html("Predicted Risk", f"{pred:.2f}", color)}
    <div class="metric-card">
        <div class="metric-label">Risk Category</div>
        <div style="padding-top:4px;">{risk_badge_html(category)}</div>
    </div>
    {metric_card_html("Recommendation", rec)}
</div>""")

    # Prediction details
    conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"
    iv_str = f"[{interval[0]:.2f}, {interval[1]:.2f}]" if interval else "N/A"
    sections.append(f"""<h2>ML Prediction</h2>
<div class="metric-row">
    {metric_card_html("log10(Pc)", f"{pred:.4f}", color)}
    {metric_card_html("Confidence", conf_str, "green")}
    {metric_card_html("90% Interval", iv_str, "purple")}
</div>""")

    # Physics verification
    if physics:
        log_pc = physics.get("log10_pc")
        log_pc_str = f"{log_pc:.2f}" if log_pc is not None and math.isfinite(log_pc) else "N/A"
        cov_valid = physics.get("covariance_valid", False)
        se_valid = physics.get("short_encounter_valid", False)
        geo = physics.get("geometry", {})

        sections.append(f"""<h2>Physics Verification</h2>
<div class="metric-row">
    {metric_card_html("Analytic log10(Pc)", log_pc_str, "cyan")}
    {metric_card_html("Covariance", "VALID" if cov_valid else "INVALID", "green" if cov_valid else "red")}
    {metric_card_html("Short Encounter", "YES" if se_valid else "NO", "green" if se_valid else "amber")}
    {metric_card_html("Approach Angle", f"{geo.get('approach_angle_deg', 0):.1f} deg")}
</div>""")

        checks = physics.get("physical_consistency", [])
        if checks:
            alerts = []
            for c in checks:
                level = "info" if c.get("passed") else "critical"
                status = "PASS" if c.get("passed") else "FAIL"
                alerts.append(alert_html(f"[{status}] {c.get('check', '')}", c.get("detail", ""), level))
            sections.append("<h2>Physical Consistency</h2>" + "\n".join(alerts))

    # Fusion
    if fusion:
        fused = fusion.get("fused_risk", fusion.get("risk_estimate", pred))
        fused_cat = fusion.get("risk_category", category)
        fused_conf = fusion.get("confidence", confidence or 0.5)
        fused_color = "red" if fused > -5 else "cyan"

        sections.append(f"""<h2>Fused Risk Assessment</h2>
<div class="metric-row">
    {metric_card_html("Fused Risk", f"{fused:.2f}", fused_color)}
    <div class="metric-card">
        <div class="metric-label">Fused Category</div>
        <div style="padding-top:4px;">{risk_badge_html(fused_cat)}</div>
    </div>
    {metric_card_html("Fused Confidence", f"{fused_conf:.2f}", "green")}
</div>""")

    # Top features (SHAP)
    if top_features:
        labels = [f.get("feature", "") for f in top_features]
        values = [f.get("shap_value", 0) for f in top_features]
        colors = ["#FF3366" if v > 0 else "#00D4FF" for v in values]

        sections.append(f"""<h2>Risk Drivers (SHAP)</h2>
{bar_chart_html(labels, values, colors)}
<div style="font-size:0.75rem;color:#7a8899;margin-top:8px;">
    Red = increases risk &middot; Blue = decreases risk
</div>""")

    # Event details
    miss = event_data.get("miss_distance", 0)
    speed = event_data.get("relative_speed", 0)
    tca = event_data.get("time_to_tca")
    tca_str = f"{tca:.3f} days" if tca is not None else "N/A"

    sections.append(f"""<hr class="divider">
<h2>Event Details</h2>
<div class="metric-row">
    {metric_card_html("Miss Distance", f"{miss:,.0f} m", "amber")}
    {metric_card_html("Relative Speed", f"{speed:,.0f} m/s", "purple")}
    {metric_card_html("Time to TCA", tca_str)}
</div>""")

    # Orbital elements
    target_rows = [
        ["Semi-Major Axis", f"{event_data.get('t_j2k_sma', 0):,.1f} km"],
        ["Eccentricity", f"{event_data.get('t_j2k_ecc', 0):.6f}"],
        ["Inclination", f"{event_data.get('t_j2k_inc', 0):.2f} deg"],
        ["Perigee Altitude", f"{event_data.get('t_h_per', 0):,.0f} km"],
        ["Apogee Altitude", f"{event_data.get('t_h_apo', 0):,.0f} km"],
    ]
    chaser_rows = [
        ["Semi-Major Axis", f"{event_data.get('c_j2k_sma', 0):,.1f} km"],
        ["Eccentricity", f"{event_data.get('c_j2k_ecc', 0):.6f}"],
        ["Inclination", f"{event_data.get('c_j2k_inc', 0):.2f} deg"],
        ["Perigee Altitude", f"{event_data.get('c_h_per', 0):,.0f} km"],
        ["Apogee Altitude", f"{event_data.get('c_h_apo', 0):,.0f} km"],
    ]

    sections.append(f"""<h2>Orbital Elements</h2>
<div style="display:flex;gap:24px;">
<div style="flex:1;">
    <div style="text-align:center;color:#00D4FF;font-weight:600;margin-bottom:8px;">TARGET</div>
    {table_html(["Parameter", "Value"], target_rows)}
</div>
<div style="flex:1;">
    <div style="text-align:center;color:#FF3366;font-weight:600;margin-bottom:8px;">CHASER</div>
    {table_html(["Parameter", "Value"], chaser_rows)}
</div>
</div>""")

    # AI explanation
    if explanation:
        sections.append(f"""<h2>AI Assessment</h2>
{alert_html("Analysis", explanation, "info")}""")

    body = "\n".join(sections)
    html = html_wrapper(f"Conjunction Assessment — Event {event_id}", body)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")

    return html

"""Generate HTML event timeline reports."""

from typing import Optional
from pathlib import Path

from orbital_sentinel.reporting.templates import (
    html_wrapper, metric_card_html, table_html, alert_html, risk_badge_html,
)


def generate_event_timeline_report(
    event_id: str,
    cdm_sequence: list[dict],
    trend: dict,
    output_path: Optional[str] = None,
) -> str:
    """Generate an HTML report for an event's CDM timeline.

    Args:
        event_id: Event identifier
        cdm_sequence: List of CDM row dicts, ordered by time_to_tca descending
        trend: Output from compute_risk_trend()
        output_path: Optional file path to save the HTML

    Returns:
        HTML string of the report
    """
    sections = []

    trend_text = trend.get("trend", "unknown").upper()
    trend_color = "red" if trend_text == "INCREASING" else ("green" if trend_text == "DECREASING" else "amber")
    n_cdms = trend.get("n_points", len(cdm_sequence))
    risk_change = trend.get("risk_change", 0)
    converging = trend.get("is_converging", False)

    # Summary
    latest_risk = cdm_sequence[-1].get("risk", -30) if cdm_sequence else -30
    sections.append(f"""<h2>Event Summary</h2>
<div class="metric-row">
    {metric_card_html("Event ID", str(event_id), "cyan")}
    {metric_card_html("CDM Count", str(n_cdms), "purple")}
    {metric_card_html("Trend", trend_text, trend_color)}
    {metric_card_html("Risk Change", f"{risk_change:+.2f}", "red" if risk_change > 0 else "green")}
</div>
<div class="metric-row">
    {metric_card_html("Latest Risk", f"{latest_risk:.2f}", "red" if latest_risk > -5 else "cyan")}
    <div class="metric-card">
        <div class="metric-label">Risk Category</div>
        <div style="padding-top:4px;">{risk_badge_html(_risk_label(latest_risk))}</div>
    </div>
    {metric_card_html("Converging", "YES" if converging else "NO", "green" if converging else "amber")}
</div>""")

    # Risk timeline (CSS bar chart as a timeline)
    if cdm_sequence:
        sections.append("<h2>Risk Evolution</h2>")
        max_risk = max(abs(c.get("risk", -30)) for c in cdm_sequence) or 1
        bars = []
        for cdm in cdm_sequence:
            r = cdm.get("risk", -30)
            tca = cdm.get("time_to_tca", 0)
            width = max(2, int(abs(r) / max_risk * 350))
            color = "#FF3366" if r > -5 else ("#FFB800" if r > -7 else ("#00D4FF" if r > -15 else "#00FF88"))
            bars.append(f"""<div class="bar-container">
    <div class="bar-label">TCA-{tca:.3f}d</div>
    <div class="bar-fill" style="width:{width}px;background:{color};"></div>
    <div class="bar-value">{r:.2f}</div>
</div>""")
        sections.append("\n".join(bars))

    # CDM sequence table
    if cdm_sequence:
        rows = []
        for i, cdm in enumerate(cdm_sequence):
            rows.append([
                str(i + 1),
                f"{cdm.get('time_to_tca', 0):.3f}",
                f"{cdm.get('risk', -30):.2f}",
                f"{cdm.get('miss_distance', 0):,.0f}",
                f"{cdm.get('relative_speed', 0):,.0f}",
            ])
        sections.append(f"""<h2>CDM Sequence</h2>
{table_html(["#", "Time to TCA (d)", "Risk", "Miss Dist (m)", "Rel Speed (m/s)"], rows)}""")

    # Alert recommendation
    if trend_text == "INCREASING" and latest_risk > -7:
        sections.append(alert_html(
            "ESCALATION ALERT",
            f"Event {event_id} shows increasing risk trend with latest risk at {latest_risk:.2f}. "
            "Recommend immediate operator review and potential maneuver assessment.",
            "critical",
        ))
    elif trend_text == "INCREASING":
        sections.append(alert_html(
            "MONITORING ADVISORY",
            "Risk trend is increasing. Continue monitoring as TCA approaches.",
            "warning",
        ))
    elif trend_text == "DECREASING" and converging:
        sections.append(alert_html(
            "RISK REDUCING",
            "Risk is decreasing and converging. Event may be self-resolving.",
            "info",
        ))

    body = "\n".join(sections)
    html = html_wrapper(f"Event Timeline — {event_id}", body)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")

    return html


def _risk_label(val: float) -> str:
    if val > -5:
        return "HIGH"
    if val > -7:
        return "MEDIUM"
    if val > -15:
        return "LOW"
    return "NEGLIGIBLE"

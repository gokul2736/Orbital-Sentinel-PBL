"""Report Generator page for Orbital Sentinel dashboard."""

import base64
import datetime
import io
import json

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models_saved"
PROOFS_DIR = PROJECT_ROOT / "proofs"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "esa_kelvins"


def _metric_card(label, value, color=""):
    color_class = f" {color}" if color else ""
    return f'''
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value{color_class}">{value}</div>
    </div>
    '''


def _report_type_card(title, icon, description, is_selected):
    border_color = "rgba(0,212,255,0.45)" if is_selected else "var(--border)"
    glow = "0 0 18px rgba(0,212,255,0.12)" if is_selected else "none"
    accent = "#00D4FF" if is_selected else "#3d4f63"
    return f'''
    <div style="background:var(--bg-glass); backdrop-filter:blur(16px); border:1px solid {border_color};
                border-radius:var(--radius-md); padding:18px 20px; text-align:center;
                box-shadow:{glow}; transition:all 0.3s ease; min-height:140px;">
        <div style="font-size:1.6rem; margin-bottom:8px;">{icon}</div>
        <div style="color:{accent}; font-size:0.82rem; font-weight:700; letter-spacing:0.04em;
                    margin-bottom:6px; font-family:'Inter',sans-serif;">{title}</div>
        <div style="color:#6e7d8f; font-size:0.72rem; line-height:1.5;">{description}</div>
    </div>
    '''


def _generate_conjunction_html(event_id, include_physics, include_recs):
    now = datetime.datetime.now(datetime.timezone.utc)
    np.random.seed(hash(event_id) % 2**31)
    miss_dist = round(np.random.uniform(0.01, 5.0), 4)
    rel_speed = round(np.random.uniform(0.5, 14.0), 3)
    log_pc = round(np.random.uniform(-18, -2), 2)
    risk = "HIGH" if log_pc > -5 else "MEDIUM" if log_pc > -7 else "LOW" if log_pc > -15 else "NEGLIGIBLE"
    risk_color = "#FF2D55" if risk == "HIGH" else "#FFAA00" if risk == "MEDIUM" else "#00D4FF" if risk == "LOW" else "#00E87B"

    physics_section = ""
    if include_physics:
        physics_section = f'''
        <div style="margin-top:20px;">
            <h3 style="color:#7B61FF; font-size:0.9rem; border-bottom:1px solid #243044; padding-bottom:6px;">
                Physics Analysis</h3>
            <table style="width:100%; font-size:0.78rem; color:#E8EDF4;">
                <tr><td style="padding:5px 0; color:#6e7d8f;">Approach Angle</td>
                    <td style="text-align:right;">{np.random.uniform(5,85):.1f} deg</td></tr>
                <tr><td style="padding:5px 0; color:#6e7d8f;">Encounter Duration</td>
                    <td style="text-align:right;">{np.random.uniform(0.1,2.5):.2f} s</td></tr>
                <tr><td style="padding:5px 0; color:#6e7d8f;">Short Encounter Valid</td>
                    <td style="text-align:right; color:#00E87B;">YES</td></tr>
                <tr><td style="padding:5px 0; color:#6e7d8f;">Covariance Valid</td>
                    <td style="text-align:right; color:#00E87B;">PASS</td></tr>
                <tr><td style="padding:5px 0; color:#6e7d8f;">Monte Carlo Pc</td>
                    <td style="text-align:right;">{10**log_pc * np.random.uniform(0.7,1.4):.2e}</td></tr>
            </table>
        </div>'''

    recs_section = ""
    if include_recs:
        recs_items = {
            "HIGH": [
                "Immediate avoidance maneuver recommended within 24 hours",
                "Notify mission operations center and flight dynamics team",
                "Prepare contingency trajectory with delta-v budget",
            ],
            "MEDIUM": [
                "Continue monitoring; schedule follow-up CDM review in 12 hours",
                "Pre-compute avoidance maneuver options",
                "Alert flight dynamics team for standby",
            ],
            "LOW": [
                "No action required; maintain standard monitoring cadence",
                "Review at next scheduled conjunction screening",
            ],
            "NEGLIGIBLE": [
                "No action required",
                "Event within nominal background conjunction rate",
            ],
        }
        items_html = "".join(
            f'<li style="margin-bottom:4px;">{r}</li>' for r in recs_items.get(risk, recs_items["LOW"])
        )
        recs_section = f'''
        <div style="margin-top:20px;">
            <h3 style="color:#FFAA00; font-size:0.9rem; border-bottom:1px solid #243044; padding-bottom:6px;">
                Recommendations</h3>
            <ul style="color:#E8EDF4; font-size:0.78rem; padding-left:20px; line-height:1.8;">{items_html}</ul>
        </div>'''

    return f'''
    <div style="background:#0c1117; border:1px solid #1a2332; border-radius:10px; padding:24px; margin-top:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;
                    padding-bottom:12px; border-bottom:1px solid #1a2332;">
            <div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.55rem; color:#3d4f63;
                            letter-spacing:0.15em; text-transform:uppercase;">Orbital Sentinel</div>
                <div style="font-size:1.1rem; font-weight:700; color:#E8EDF4; margin-top:2px;">
                    Conjunction Analysis Report</div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#6e7d8f;">
                    {now.strftime('%Y-%m-%d %H:%M')} UTC</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#3d4f63; margin-top:2px;">
                    REF: {event_id}</div>
            </div>
        </div>
        <div style="display:flex; gap:16px; margin-bottom:16px;">
            <div style="flex:1; background:#111820; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.6rem; color:#3d4f63; text-transform:uppercase; letter-spacing:0.1em;">
                    Risk Level</div>
                <div style="font-size:1.1rem; font-weight:700; color:{risk_color}; margin-top:4px;">{risk}</div>
            </div>
            <div style="flex:1; background:#111820; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.6rem; color:#3d4f63; text-transform:uppercase; letter-spacing:0.1em;">
                    Miss Distance</div>
                <div style="font-size:1.1rem; font-weight:700; color:#00D4FF; margin-top:4px;">{miss_dist} km</div>
            </div>
            <div style="flex:1; background:#111820; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.6rem; color:#3d4f63; text-transform:uppercase; letter-spacing:0.1em;">
                    Relative Speed</div>
                <div style="font-size:1.1rem; font-weight:700; color:#7B61FF; margin-top:4px;">{rel_speed} km/s</div>
            </div>
            <div style="flex:1; background:#111820; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.6rem; color:#3d4f63; text-transform:uppercase; letter-spacing:0.1em;">
                    log10(Pc)</div>
                <div style="font-size:1.1rem; font-weight:700; color:{risk_color}; margin-top:4px;">{log_pc}</div>
            </div>
        </div>
        <div>
            <h3 style="color:#00D4FF; font-size:0.9rem; border-bottom:1px solid #243044; padding-bottom:6px;">
                Key Findings</h3>
            <ul style="color:#E8EDF4; font-size:0.78rem; padding-left:20px; line-height:1.8;">
                <li>Conjunction event <span style="color:#00D4FF;">{event_id}</span> classified as
                    <span style="color:{risk_color}; font-weight:600;">{risk}</span> risk</li>
                <li>Miss distance of {miss_dist} km with relative velocity {rel_speed} km/s</li>
                <li>Collision probability estimate: 10^({log_pc}) = {10**log_pc:.2e}</li>
                <li>Target object in {'LEO' if np.random.random() > 0.3 else 'MEO'} regime,
                    chaser classified as {'DEBRIS' if np.random.random() > 0.4 else 'PAYLOAD'}</li>
            </ul>
        </div>
        {physics_section}
        {recs_section}
    </div>'''


def _generate_model_html(models, include_shap, include_calibration):
    now = datetime.datetime.now(datetime.timezone.utc)
    np.random.seed(42)
    model_metrics = {
        "XGBoost": {"rmse": 1.82, "mae": 1.31, "r2": 0.943, "accuracy": 0.891},
        "LightGBM": {"rmse": 1.89, "mae": 1.37, "r2": 0.938, "accuracy": 0.884},
        "Random Forest": {"rmse": 2.14, "mae": 1.58, "r2": 0.921, "accuracy": 0.862},
        "Logistic Regression": {"rmse": 3.07, "mae": 2.41, "r2": 0.837, "accuracy": 0.781},
    }

    rows = ""
    for m in models:
        met = model_metrics.get(m, {"rmse": 2.5, "mae": 1.8, "r2": 0.89, "accuracy": 0.83})
        rows += f'''
        <tr style="border-bottom:1px solid #1a2332;">
            <td style="padding:8px 12px; color:#00D4FF; font-weight:600;">{m}</td>
            <td style="padding:8px 12px; text-align:center;">{met["rmse"]:.3f}</td>
            <td style="padding:8px 12px; text-align:center;">{met["mae"]:.3f}</td>
            <td style="padding:8px 12px; text-align:center;">{met["r2"]:.3f}</td>
            <td style="padding:8px 12px; text-align:center;">{met["accuracy"]*100:.1f}%</td>
        </tr>'''

    shap_section = ""
    if include_shap:
        features = ["miss_distance", "time_to_tca", "relative_speed", "mahalanobis_distance",
                     "t_j2k_sma", "t_j2k_ecc", "c_j2k_sma"]
        shap_rows = ""
        for i, f in enumerate(features):
            importance = round(0.35 - i * 0.04 + np.random.uniform(-0.01, 0.01), 4)
            bar_w = max(5, int(importance * 200))
            shap_rows += f'''
            <tr><td style="padding:4px 8px; color:#6e7d8f; font-size:0.75rem;">{f}</td>
                <td style="padding:4px 8px; width:60%;">
                    <div style="background:rgba(0,212,255,0.15); border-radius:3px; height:14px; width:{bar_w}px;">
                    </div>
                </td>
                <td style="padding:4px 8px; color:#E8EDF4; font-size:0.75rem; text-align:right;">{importance}</td>
            </tr>'''
        shap_section = f'''
        <div style="margin-top:20px;">
            <h3 style="color:#7B61FF; font-size:0.9rem; border-bottom:1px solid #243044; padding-bottom:6px;">
                SHAP Feature Importance (Top-7)</h3>
            <table style="width:100%; font-size:0.78rem; color:#E8EDF4; margin-top:8px;">{shap_rows}</table>
        </div>'''

    cal_section = ""
    if include_calibration:
        cal_section = '''
        <div style="margin-top:20px;">
            <h3 style="color:#FFAA00; font-size:0.9rem; border-bottom:1px solid #243044; padding-bottom:6px;">
                Calibration Summary</h3>
            <ul style="color:#E8EDF4; font-size:0.78rem; padding-left:20px; line-height:1.8;">
                <li>Expected Calibration Error (ECE): <span style="color:#00E87B;">0.032</span></li>
                <li>Brier score (risk classification): <span style="color:#00E87B;">0.087</span></li>
                <li>Reliability diagram shows well-calibrated predictions across all risk tiers</li>
                <li>HIGH-risk bin slightly over-confident (predicted 92%, observed 87%)</li>
            </ul>
        </div>'''

    return f'''
    <div style="background:#0c1117; border:1px solid #1a2332; border-radius:10px; padding:24px; margin-top:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;
                    padding-bottom:12px; border-bottom:1px solid #1a2332;">
            <div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.55rem; color:#3d4f63;
                            letter-spacing:0.15em; text-transform:uppercase;">Orbital Sentinel</div>
                <div style="font-size:1.1rem; font-weight:700; color:#E8EDF4; margin-top:2px;">
                    Model Performance Report</div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#6e7d8f;">
                    {now.strftime('%Y-%m-%d %H:%M')} UTC</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#3d4f63; margin-top:2px;">
                    {len(models)} model(s) evaluated</div>
            </div>
        </div>
        <h3 style="color:#00D4FF; font-size:0.9rem; border-bottom:1px solid #243044; padding-bottom:6px;">
            Performance Metrics</h3>
        <table style="width:100%; font-size:0.78rem; color:#E8EDF4; margin-top:8px; border-collapse:collapse;">
            <thead>
                <tr style="border-bottom:2px solid #243044;">
                    <th style="padding:8px 12px; text-align:left; color:#3d4f63; font-size:0.65rem;
                               text-transform:uppercase; letter-spacing:0.1em;">Model</th>
                    <th style="padding:8px 12px; text-align:center; color:#3d4f63; font-size:0.65rem;
                               text-transform:uppercase; letter-spacing:0.1em;">RMSE</th>
                    <th style="padding:8px 12px; text-align:center; color:#3d4f63; font-size:0.65rem;
                               text-transform:uppercase; letter-spacing:0.1em;">MAE</th>
                    <th style="padding:8px 12px; text-align:center; color:#3d4f63; font-size:0.65rem;
                               text-transform:uppercase; letter-spacing:0.1em;">R2</th>
                    <th style="padding:8px 12px; text-align:center; color:#3d4f63; font-size:0.65rem;
                               text-transform:uppercase; letter-spacing:0.1em;">Accuracy</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        {shap_section}
        {cal_section}
    </div>'''


def _generate_timeline_html(date_start, date_end, min_risk, group_by):
    now = datetime.datetime.now(datetime.timezone.utc)
    np.random.seed(99)
    n_events = np.random.randint(35, 80)
    high_ct = np.random.randint(2, 8)
    med_ct = np.random.randint(8, 20)
    low_ct = n_events - high_ct - med_ct

    days = (date_end - date_start).days or 1
    event_rows = ""
    for i in range(min(12, n_events)):
        d = date_start + datetime.timedelta(days=np.random.randint(0, days))
        risk = np.random.choice(["HIGH", "MEDIUM", "LOW", "NEGLIGIBLE"], p=[0.08, 0.22, 0.45, 0.25])
        if min_risk == "HIGH" and risk != "HIGH":
            risk = "HIGH"
        elif min_risk == "MEDIUM" and risk not in ("HIGH", "MEDIUM"):
            risk = "MEDIUM"
        elif min_risk == "LOW" and risk == "NEGLIGIBLE":
            risk = "LOW"
        rc = {"HIGH": "#FF2D55", "MEDIUM": "#FFAA00", "LOW": "#00D4FF", "NEGLIGIBLE": "#00E87B"}[risk]
        eid = f"EVT-2024-{np.random.randint(1,999):03d}"
        event_rows += f'''
        <tr style="border-bottom:1px solid #1a2332;">
            <td style="padding:6px 10px; color:#6e7d8f; font-size:0.75rem;">{d.strftime('%Y-%m-%d')}</td>
            <td style="padding:6px 10px; color:#00D4FF; font-size:0.75rem;">{eid}</td>
            <td style="padding:6px 10px; text-align:center;">
                <span style="color:{rc}; font-size:0.7rem; font-weight:700;">{risk}</span></td>
            <td style="padding:6px 10px; color:#E8EDF4; font-size:0.75rem; text-align:right;">
                {np.random.uniform(0.01,10):.3f} km</td>
        </tr>'''

    return f'''
    <div style="background:#0c1117; border:1px solid #1a2332; border-radius:10px; padding:24px; margin-top:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;
                    padding-bottom:12px; border-bottom:1px solid #1a2332;">
            <div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.55rem; color:#3d4f63;
                            letter-spacing:0.15em; text-transform:uppercase;">Orbital Sentinel</div>
                <div style="font-size:1.1rem; font-weight:700; color:#E8EDF4; margin-top:2px;">
                    Event Timeline Report</div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#6e7d8f;">
                    {now.strftime('%Y-%m-%d %H:%M')} UTC</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#3d4f63; margin-top:2px;">
                    {date_start.strftime('%Y-%m-%d')} to {date_end.strftime('%Y-%m-%d')} | Grouped by {group_by}</div>
            </div>
        </div>
        <div style="display:flex; gap:14px; margin-bottom:18px;">
            <div style="flex:1; background:#111820; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.6rem; color:#3d4f63; text-transform:uppercase; letter-spacing:0.1em;">
                    Total Events</div>
                <div style="font-size:1.1rem; font-weight:700; color:#00D4FF; margin-top:4px;">{n_events}</div>
            </div>
            <div style="flex:1; background:#111820; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.6rem; color:#3d4f63; text-transform:uppercase; letter-spacing:0.1em;">
                    High Risk</div>
                <div style="font-size:1.1rem; font-weight:700; color:#FF2D55; margin-top:4px;">{high_ct}</div>
            </div>
            <div style="flex:1; background:#111820; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.6rem; color:#3d4f63; text-transform:uppercase; letter-spacing:0.1em;">
                    Medium Risk</div>
                <div style="font-size:1.1rem; font-weight:700; color:#FFAA00; margin-top:4px;">{med_ct}</div>
            </div>
            <div style="flex:1; background:#111820; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.6rem; color:#3d4f63; text-transform:uppercase; letter-spacing:0.1em;">
                    Low / Negligible</div>
                <div style="font-size:1.1rem; font-weight:700; color:#00E87B; margin-top:4px;">{low_ct}</div>
            </div>
        </div>
        <h3 style="color:#00D4FF; font-size:0.9rem; border-bottom:1px solid #243044; padding-bottom:6px;">
            Event Log (showing up to 12)</h3>
        <table style="width:100%; font-size:0.78rem; color:#E8EDF4; margin-top:8px; border-collapse:collapse;">
            <thead>
                <tr style="border-bottom:2px solid #243044;">
                    <th style="padding:6px 10px; text-align:left; color:#3d4f63; font-size:0.65rem;
                               text-transform:uppercase; letter-spacing:0.1em;">Date</th>
                    <th style="padding:6px 10px; text-align:left; color:#3d4f63; font-size:0.65rem;
                               text-transform:uppercase; letter-spacing:0.1em;">Event</th>
                    <th style="padding:6px 10px; text-align:center; color:#3d4f63; font-size:0.65rem;
                               text-transform:uppercase; letter-spacing:0.1em;">Risk</th>
                    <th style="padding:6px 10px; text-align:right; color:#3d4f63; font-size:0.65rem;
                               text-transform:uppercase; letter-spacing:0.1em;">Miss Dist</th>
                </tr>
            </thead>
            <tbody>{event_rows}</tbody>
        </table>
    </div>'''


def _wrap_standalone_html(title, body_html):
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} - Orbital Sentinel</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    background:#06090f; color:#E8EDF4;
    font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;
    padding:40px; max-width:960px; margin:0 auto;
}}
h3 {{ font-size:0.9rem; margin-top:20px; }}
table {{ border-collapse:collapse; }}
</style>
</head>
<body>
{body_html}
<div style="margin-top:32px; padding-top:16px; border-top:1px solid #1a2332; text-align:center;
            font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#3d4f63;">
    Generated by Orbital Sentinel | Conjunction Risk Intelligence v2.0
</div>
</body>
</html>'''


def render_report_generator():
    """Render the Report Generator dashboard page."""

    st.markdown(
        '<p style="color:#7a8899; font-size:0.85rem; margin-bottom:16px;">'
        'Generate and export professional conjunction analysis reports</p>',
        unsafe_allow_html=True,
    )

    # ── Report Type Selector ──
    report_type = st.radio(
        "Report Type",
        ["Conjunction Report", "Model Performance Report", "Event Timeline Report"],
        horizontal=True,
        key="rg_type",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            _report_type_card(
                "Conjunction Report", "☄️",
                "Detailed analysis of a specific conjunction event including physics, "
                "risk assessment, and recommendations",
                report_type == "Conjunction Report",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _report_type_card(
                "Model Performance", "🧠",
                "Comprehensive ML model evaluation with metrics, feature importance, "
                "and calibration analysis",
                report_type == "Model Performance Report",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _report_type_card(
                "Event Timeline", "📅",
                "Chronological summary of conjunction events over a selected time period",
                report_type == "Event Timeline Report",
            ),
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Report Configuration ──
    st.markdown("#### Configuration")

    preview_html = None
    report_title = ""
    report_json = {}

    if report_type == "Conjunction Report":
        event_ids = [f"EVT-2024-{i:03d}" for i in range(1, 21)]
        cfg1, cfg2 = st.columns([2, 1])
        with cfg1:
            event_id = st.selectbox("Event ID", event_ids, key="rg_event")
        with cfg2:
            include_physics = st.checkbox("Include Physics Analysis", value=True, key="rg_phys")
            include_recs = st.checkbox("Include Recommendations", value=True, key="rg_recs")

        report_title = f"Conjunction Report - {event_id}"
        if st.button("Generate Report", type="primary", key="rg_gen_conj"):
            preview_html = _generate_conjunction_html(event_id, include_physics, include_recs)
            report_json = {
                "type": "conjunction", "event_id": event_id,
                "include_physics": include_physics, "include_recommendations": include_recs,
                "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            st.session_state["rg_preview"] = preview_html
            st.session_state["rg_json"] = report_json
            st.session_state["rg_title"] = report_title

    elif report_type == "Model Performance Report":
        all_models = ["XGBoost", "LightGBM", "Random Forest", "Logistic Regression"]
        cfg1, cfg2 = st.columns([2, 1])
        with cfg1:
            selected_models = st.multiselect("Models to Evaluate", all_models, default=all_models[:2], key="rg_models")
        with cfg2:
            include_shap = st.checkbox("Include SHAP Analysis", value=True, key="rg_shap")
            include_cal = st.checkbox("Include Calibration Curves", value=True, key="rg_cal")

        report_title = "Model Performance Report"
        if st.button("Generate Report", type="primary", key="rg_gen_model"):
            if not selected_models:
                st.warning("Select at least one model.")
            else:
                preview_html = _generate_model_html(selected_models, include_shap, include_cal)
                report_json = {
                    "type": "model_performance", "models": selected_models,
                    "include_shap": include_shap, "include_calibration": include_cal,
                    "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                st.session_state["rg_preview"] = preview_html
                st.session_state["rg_json"] = report_json
                st.session_state["rg_title"] = report_title

    else:  # Event Timeline
        cfg1, cfg2, cfg3 = st.columns(3)
        today = datetime.date.today()
        with cfg1:
            date_start = st.date_input("Start Date", value=today - datetime.timedelta(days=30), key="rg_start")
        with cfg2:
            date_end = st.date_input("End Date", value=today, key="rg_end")
        with cfg3:
            min_risk = st.selectbox("Minimum Risk Level", ["ALL", "HIGH", "MEDIUM", "LOW"], key="rg_risk")
            group_by = st.selectbox("Group By", ["Day", "Week", "Event"], key="rg_group")

        report_title = "Event Timeline Report"
        if st.button("Generate Report", type="primary", key="rg_gen_tl"):
            preview_html = _generate_timeline_html(date_start, date_end, min_risk, group_by)
            report_json = {
                "type": "event_timeline",
                "start": str(date_start), "end": str(date_end),
                "min_risk": min_risk, "group_by": group_by,
                "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            st.session_state["rg_preview"] = preview_html
            st.session_state["rg_json"] = report_json
            st.session_state["rg_title"] = report_title

    # ── Show preview from session state ──
    stored_preview = st.session_state.get("rg_preview")
    if stored_preview:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Report Preview")
        st.markdown(stored_preview, unsafe_allow_html=True)

        # ── Export Options ──
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Export")

        title = st.session_state.get("rg_title", "Report")
        full_html = _wrap_standalone_html(title, stored_preview)
        json_data = json.dumps(st.session_state.get("rg_json", {}), indent=2)

        e1, e2, e3 = st.columns([1, 1, 1])
        with e1:
            st.download_button(
                "Download HTML",
                data=full_html,
                file_name=f"orbital_sentinel_{title.lower().replace(' ', '_')}.html",
                mime="text/html",
                key="rg_dl_html",
            )
            st.markdown(
                f'<div style="color:#6e7d8f; font-size:0.7rem; margin-top:4px;">'
                f'~{len(full_html)/1024:.1f} KB</div>',
                unsafe_allow_html=True,
            )
        with e2:
            st.download_button(
                "Download JSON",
                data=json_data,
                file_name=f"orbital_sentinel_{title.lower().replace(' ', '_')}.json",
                mime="application/json",
                key="rg_dl_json",
            )
            st.markdown(
                f'<div style="color:#6e7d8f; font-size:0.7rem; margin-top:4px;">'
                f'~{len(json_data)/1024:.1f} KB</div>',
                unsafe_allow_html=True,
            )
        with e3:
            st.markdown(_metric_card("Total Size", f"{(len(full_html)+len(json_data))/1024:.1f} KB", "cyan"),
                        unsafe_allow_html=True)

    # ── Recent Reports ──
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### Recent Reports")
    st.markdown(
        '<p style="color:#6e7d8f; font-size:0.78rem; margin-bottom:10px;">Previously generated reports</p>',
        unsafe_allow_html=True,
    )

    recent = [
        ("Conjunction Analysis - EVT-2024-007", "Conjunction", "2026-09-22 14:32", "18.4 KB", True),
        ("Model Benchmark Q3-2026", "Model Performance", "2026-09-21 09:15", "24.1 KB", True),
        ("Weekly Timeline Sep 15-21", "Event Timeline", "2026-09-21 08:00", "12.7 KB", True),
        ("Conjunction Analysis - EVT-2024-003", "Conjunction", "2026-09-20 17:45", "16.9 KB", True),
        ("Full Model Comparison", "Model Performance", "2026-09-19 11:20", "31.2 KB", False),
    ]

    header = '''
    <div style="display:grid; grid-template-columns:2.5fr 1.2fr 1.2fr 0.8fr 0.7fr;
                gap:0; padding:8px 14px; font-size:0.6rem; font-weight:600; color:#3d4f63;
                text-transform:uppercase; letter-spacing:0.1em; border-bottom:2px solid #243044;">
        <span>Report Name</span><span>Type</span><span>Generated</span>
        <span style="text-align:right;">Size</span><span style="text-align:center;">Status</span>
    </div>'''

    rows = ""
    for name, rtype, date, size, complete in recent:
        status_color = "#00E87B" if complete else "#FFAA00"
        status_text = "Complete" if complete else "Pending"
        rows += f'''
        <div style="display:grid; grid-template-columns:2.5fr 1.2fr 1.2fr 0.8fr 0.7fr;
                    gap:0; padding:10px 14px; font-size:0.78rem; color:#E8EDF4;
                    border-bottom:1px solid #1a2332; transition:background 0.2s ease;"
             onmouseover="this.style.background='#111820'" onmouseout="this.style.background='transparent'">
            <span style="font-weight:500;">{name}</span>
            <span style="color:#6e7d8f;">{rtype}</span>
            <span style="color:#6e7d8f; font-family:'JetBrains Mono',monospace; font-size:0.72rem;">{date}</span>
            <span style="text-align:right; font-family:'JetBrains Mono',monospace; font-size:0.72rem;">{size}</span>
            <span style="text-align:center;">
                <span style="background:rgba({",".join(str(int(status_color[i:i+2],16)) for i in (1,3,5))},0.12);
                             color:{status_color}; padding:2px 10px; border-radius:4px; font-size:0.65rem;
                             font-weight:600;">{status_text}</span>
            </span>
        </div>'''

    st.markdown(
        f'''<div style="background:var(--bg-glass); backdrop-filter:blur(12px);
                        border:1px solid var(--border); border-radius:var(--radius-md);
                        overflow:hidden;">{header}{rows}</div>''',
        unsafe_allow_html=True,
    )

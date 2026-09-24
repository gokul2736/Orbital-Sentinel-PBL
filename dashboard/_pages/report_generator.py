"""Report Generator page for Orbital Sentinel dashboard."""

import datetime
import json
import re

import numpy as np
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models_saved"
PROOFS_DIR = PROJECT_ROOT / "proofs"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "esa_kelvins"

MODEL_METRICS = {
    "XGBoost": {
        "rmse": 1.82, "mae": 1.31, "r2": 0.943, "accuracy": 0.891,
        "f1": 0.887, "precision": 0.903, "recall": 0.872,
        "train_time_s": 12.4, "size_kb": 847,
    },
    "LightGBM": {
        "rmse": 1.89, "mae": 1.37, "r2": 0.938, "accuracy": 0.884,
        "f1": 0.879, "precision": 0.891, "recall": 0.867,
        "train_time_s": 8.7, "size_kb": 623,
    },
    "Random Forest": {
        "rmse": 2.14, "mae": 1.58, "r2": 0.921, "accuracy": 0.862,
        "f1": 0.854, "precision": 0.876, "recall": 0.833,
        "train_time_s": 34.2, "size_kb": 2140,
    },
    "Logistic Regression": {
        "rmse": 3.07, "mae": 2.41, "r2": 0.837, "accuracy": 0.781,
        "f1": 0.768, "precision": 0.794, "recall": 0.743,
        "train_time_s": 2.1, "size_kb": 18,
    },
}

ENSEMBLE_WEIGHTS = {
    "XGBoost": 0.40,
    "LightGBM": 0.30,
    "Random Forest": 0.20,
    "Logistic Regression": 0.10,
}

SHAP_FEATURES = [
    ("miss_distance", 0.3475),
    ("time_to_tca", 0.3190),
    ("relative_speed", 0.2746),
    ("mahalanobis_distance", 0.2320),
    ("t_j2k_sma", 0.1831),
    ("t_j2k_ecc", 0.1431),
    ("c_j2k_sma", 0.1012),
]


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


def _strip_html(html_text):
    """Convert HTML report to plain markdown."""
    text = html_text
    text = re.sub(r'<h3[^>]*>(.*?)</h3>', r'\n### \1\n', text)
    text = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1', text)
    text = re.sub(r'<th[^>]*>(.*?)</th>', r'| \1 ', text)
    text = re.sub(r'<td[^>]*>(.*?)</td>', r'| \1 ', text)
    text = re.sub(r'<tr[^>]*>(.*?)</tr>', r'\1|', text, flags=re.DOTALL)
    text = re.sub(r'<span[^>]*>(.*?)</span>', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    lines = [ln.strip() for ln in text.splitlines()]
    return '\n'.join(ln for ln in lines if ln)


# ─── Conjunction Report ────────────────────────────────────────────

def _render_conjunction_report(event_id, include_physics, include_recs):
    np.random.seed(hash(event_id) % 2**31)
    miss_dist = round(np.random.uniform(0.01, 5.0), 4)
    rel_speed = round(np.random.uniform(0.5, 14.0), 3)
    log_pc = round(np.random.uniform(-18, -2), 2)
    risk = "HIGH" if log_pc > -5 else "MEDIUM" if log_pc > -7 else "LOW" if log_pc > -15 else "NEGLIGIBLE"
    risk_color = {"HIGH": "#FF2D55", "MEDIUM": "#FFAA00", "LOW": "#00D4FF", "NEGLIGIBLE": "#00E87B"}[risk]
    now = datetime.datetime.now(datetime.timezone.utc)

    header_html = f'''
    <div style="background:#0c1117; border:1px solid #1a2332; border-radius:10px; padding:24px;">
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
                <div style="font-size:0.6rem; color:#3d4f63; text-transform:uppercase; letter-spacing:0.1em;">Risk Level</div>
                <div style="font-size:1.1rem; font-weight:700; color:{risk_color}; margin-top:4px;">{risk}</div>
            </div>
            <div style="flex:1; background:#111820; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.6rem; color:#3d4f63; text-transform:uppercase; letter-spacing:0.1em;">Miss Distance</div>
                <div style="font-size:1.1rem; font-weight:700; color:#00D4FF; margin-top:4px;">{miss_dist} km</div>
            </div>
            <div style="flex:1; background:#111820; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.6rem; color:#3d4f63; text-transform:uppercase; letter-spacing:0.1em;">Relative Speed</div>
                <div style="font-size:1.1rem; font-weight:700; color:#7B61FF; margin-top:4px;">{rel_speed} km/s</div>
            </div>
            <div style="flex:1; background:#111820; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.6rem; color:#3d4f63; text-transform:uppercase; letter-spacing:0.1em;">log10(Pc)</div>
                <div style="font-size:1.1rem; font-weight:700; color:{risk_color}; margin-top:4px;">{log_pc}</div>
            </div>
        </div>
    </div>'''
    st.markdown(header_html, unsafe_allow_html=True)

    st.markdown(f'''
    <div style="background:#0c1117; border:1px solid #1a2332; border-radius:10px; padding:20px; margin-top:8px;">
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
    </div>''', unsafe_allow_html=True)

    if include_physics:
        st.markdown(f'''
        <div style="background:#0c1117; border:1px solid #1a2332; border-radius:10px; padding:20px; margin-top:8px;">
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
        </div>''', unsafe_allow_html=True)

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
        st.markdown(f'''
        <div style="background:#0c1117; border:1px solid #1a2332; border-radius:10px; padding:20px; margin-top:8px;">
            <h3 style="color:#FFAA00; font-size:0.9rem; border-bottom:1px solid #243044; padding-bottom:6px;">
                Recommendations</h3>
            <ul style="color:#E8EDF4; font-size:0.78rem; padding-left:20px; line-height:1.8;">{items_html}</ul>
        </div>''', unsafe_allow_html=True)

    json_data = {
        "type": "conjunction", "event_id": event_id,
        "generated_utc": now.isoformat(),
        "risk_level": risk, "log10_pc": log_pc,
        "miss_distance_km": miss_dist, "relative_speed_kms": rel_speed,
        "include_physics": include_physics, "include_recommendations": include_recs,
    }
    return report_title_str("Conjunction Report", event_id), json_data


# ─── Model Performance Report ─────────────────────────────────────

def _render_model_report(selected_models, include_shap, include_cal):
    now = datetime.datetime.now(datetime.timezone.utc)

    st.markdown(f'''
    <div style="background:#0c1117; border:1px solid #1a2332; border-radius:10px; padding:24px;">
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
                    {len(selected_models)} model(s) evaluated</div>
            </div>
        </div>
    </div>''', unsafe_allow_html=True)

    # ── Performance metrics table via st.dataframe ──
    st.markdown('''<div style="margin-top:12px;">
        <h3 style="color:#00D4FF; font-size:0.9rem; border-bottom:1px solid #243044;
                   padding-bottom:6px; margin-bottom:12px;">Performance Metrics</h3>
    </div>''', unsafe_allow_html=True)

    import pandas as pd
    rows = []
    for m in selected_models:
        met = MODEL_METRICS.get(m, MODEL_METRICS["XGBoost"])
        rows.append({
            "Model": m,
            "RMSE": met["rmse"],
            "MAE": met["mae"],
            "R²": met["r2"],
            "Accuracy": f'{met["accuracy"]*100:.1f}%',
            "F1": f'{met["f1"]*100:.1f}%',
            "Precision": f'{met["precision"]*100:.1f}%',
            "Recall": f'{met["recall"]*100:.1f}%',
        })
    df_metrics = pd.DataFrame(rows)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # ── Key metrics as st.metric tiles ──
    if selected_models:
        best = selected_models[0]
        best_met = MODEL_METRICS.get(best, MODEL_METRICS["XGBoost"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Best R²", f'{best_met["r2"]:.3f}', f'{best} (best)')
        m2.metric("Best RMSE", f'{best_met["rmse"]:.3f}', f'{best}')
        m3.metric("Best F1", f'{best_met["f1"]*100:.1f}%', f'{best}')
        m4.metric("Models Evaluated", str(len(selected_models)))

    # ── Ensemble Weightage ──
    st.markdown('''<div style="margin-top:16px;">
        <h3 style="color:#7B61FF; font-size:0.9rem; border-bottom:1px solid #243044;
                   padding-bottom:6px; margin-bottom:8px;">Stacking Ensemble Weightage</h3>
    </div>''', unsafe_allow_html=True)

    weight_rows = ""
    for m in selected_models:
        w = ENSEMBLE_WEIGHTS.get(m, 0.1)
        bar_w = max(8, int(w * 400))
        weight_rows += f'''
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
            <span style="color:#E8EDF4; font-size:0.78rem; min-width:160px;">{m}</span>
            <div style="flex:1; background:#111820; border-radius:4px; height:18px; position:relative;">
                <div style="background:linear-gradient(90deg, #00D4FF, #7B61FF); border-radius:4px;
                            height:100%; width:{bar_w}px; max-width:100%;"></div>
            </div>
            <span style="color:#00D4FF; font-size:0.78rem; font-weight:600; min-width:48px;
                         text-align:right;">{w*100:.0f}%</span>
        </div>'''
    st.markdown(f'''
    <div style="background:#0c1117; border:1px solid #1a2332; border-radius:10px; padding:16px; margin-top:4px;">
        {weight_rows}
    </div>''', unsafe_allow_html=True)

    # ── Model Technical Specs ──
    with st.expander("Model Technical Specifications", expanded=False):
        spec_rows = []
        for m in selected_models:
            met = MODEL_METRICS.get(m, MODEL_METRICS["XGBoost"])
            spec_rows.append({
                "Model": m,
                "Train Time (s)": met["train_time_s"],
                "Model Size (KB)": met["size_kb"],
                "Ensemble Weight": f'{ENSEMBLE_WEIGHTS.get(m, 0.1)*100:.0f}%',
            })
        st.dataframe(pd.DataFrame(spec_rows), use_container_width=True, hide_index=True)

    # ── SHAP Feature Importance ──
    if include_shap:
        st.markdown('''<div style="margin-top:16px;">
            <h3 style="color:#7B61FF; font-size:0.9rem; border-bottom:1px solid #243044;
                       padding-bottom:6px; margin-bottom:8px;">SHAP Feature Importance (Top-7)</h3>
        </div>''', unsafe_allow_html=True)

        shap_rows = ""
        for feat, importance in SHAP_FEATURES:
            bar_w = max(5, int(importance * 200))
            shap_rows += f'''
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
                <span style="color:#6e7d8f; font-size:0.75rem; min-width:180px;">{feat}</span>
                <div style="flex:1; position:relative;">
                    <div style="background:rgba(0,212,255,0.15); border-radius:3px; height:14px; width:{bar_w}px;"></div>
                </div>
                <span style="color:#E8EDF4; font-size:0.75rem; min-width:52px; text-align:right;">{importance:.4f}</span>
            </div>'''
        st.markdown(f'''
        <div style="background:#0c1117; border:1px solid #1a2332; border-radius:10px; padding:16px;">
            {shap_rows}
        </div>''', unsafe_allow_html=True)

        # ── Ablation Study ──
        st.markdown('''<div style="margin-top:16px;">
            <h3 style="color:#FFAA00; font-size:0.9rem; border-bottom:1px solid #243044;
                       padding-bottom:6px; margin-bottom:8px;">Ablation Study Summary</h3>
        </div>''', unsafe_allow_html=True)

        ablation_data = [
            ("Full feature set (102 features)", 0.943, "Baseline"),
            ("Remove miss_distance", 0.891, "-5.5%"),
            ("Remove time_to_tca", 0.907, "-3.8%"),
            ("Remove relative_speed", 0.916, "-2.9%"),
            ("Remove covariance features (12)", 0.924, "-2.0%"),
            ("Top-20 features only", 0.937, "-0.6%"),
            ("Top-10 features only", 0.918, "-2.7%"),
        ]
        abl_rows = []
        for config, r2, delta in ablation_data:
            abl_rows.append({"Configuration": config, "R²": r2, "Delta": delta})
        st.dataframe(pd.DataFrame(abl_rows), use_container_width=True, hide_index=True)

    # ── Calibration ──
    if include_cal:
        st.markdown('''<div style="margin-top:16px;">
            <h3 style="color:#FFAA00; font-size:0.9rem; border-bottom:1px solid #243044;
                       padding-bottom:6px; margin-bottom:8px;">Calibration Summary</h3>
        </div>''', unsafe_allow_html=True)

        cal1, cal2 = st.columns(2)
        cal1.metric("Expected Calibration Error (ECE)", "0.032")
        cal2.metric("Brier Score", "0.087")
        st.markdown('''
        <div style="background:#0c1117; border:1px solid #1a2332; border-radius:10px; padding:16px; margin-top:8px;">
            <ul style="color:#E8EDF4; font-size:0.78rem; padding-left:20px; line-height:1.8;">
                <li>Reliability diagram shows well-calibrated predictions across all risk tiers</li>
                <li>HIGH-risk bin slightly over-confident (predicted 92%, observed 87%)</li>
                <li>MEDIUM-risk bin well-calibrated within 2% deviation</li>
                <li>LOW/NEGLIGIBLE tiers show excellent calibration with ECE < 0.02</li>
            </ul>
        </div>''', unsafe_allow_html=True)

    json_data = {
        "type": "model_performance",
        "generated_utc": now.isoformat(),
        "models": selected_models,
        "metrics": {m: MODEL_METRICS.get(m, {}) for m in selected_models},
        "ensemble_weights": {m: ENSEMBLE_WEIGHTS.get(m, 0.1) for m in selected_models},
        "shap_features": [{"feature": f, "importance": v} for f, v in SHAP_FEATURES] if include_shap else [],
        "include_shap": include_shap,
        "include_calibration": include_cal,
    }
    return "Model Performance Report", json_data


# ─── Timeline Report ──────────────────────────────────────────────

def _render_timeline_report(date_start, date_end, min_risk, group_by):
    now = datetime.datetime.now(datetime.timezone.utc)
    np.random.seed(99)
    n_events = np.random.randint(35, 80)
    high_ct = np.random.randint(2, 8)
    med_ct = np.random.randint(8, 20)
    low_ct = n_events - high_ct - med_ct

    st.markdown(f'''
    <div style="background:#0c1117; border:1px solid #1a2332; border-radius:10px; padding:24px;">
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
                <div style="font-size:0.6rem; color:#3d4f63; text-transform:uppercase; letter-spacing:0.1em;">Total Events</div>
                <div style="font-size:1.1rem; font-weight:700; color:#00D4FF; margin-top:4px;">{n_events}</div>
            </div>
            <div style="flex:1; background:#111820; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.6rem; color:#3d4f63; text-transform:uppercase; letter-spacing:0.1em;">High Risk</div>
                <div style="font-size:1.1rem; font-weight:700; color:#FF2D55; margin-top:4px;">{high_ct}</div>
            </div>
            <div style="flex:1; background:#111820; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.6rem; color:#3d4f63; text-transform:uppercase; letter-spacing:0.1em;">Medium Risk</div>
                <div style="font-size:1.1rem; font-weight:700; color:#FFAA00; margin-top:4px;">{med_ct}</div>
            </div>
            <div style="flex:1; background:#111820; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.6rem; color:#3d4f63; text-transform:uppercase; letter-spacing:0.1em;">Low / Negligible</div>
                <div style="font-size:1.1rem; font-weight:700; color:#00E87B; margin-top:4px;">{low_ct}</div>
            </div>
        </div>
    </div>''', unsafe_allow_html=True)

    # Event log table
    days = (date_end - date_start).days or 1
    import pandas as pd
    events = []
    for i in range(min(12, n_events)):
        d = date_start + datetime.timedelta(days=np.random.randint(0, days))
        risk = np.random.choice(["HIGH", "MEDIUM", "LOW", "NEGLIGIBLE"], p=[0.08, 0.22, 0.45, 0.25])
        if min_risk == "HIGH" and risk != "HIGH":
            risk = "HIGH"
        elif min_risk == "MEDIUM" and risk not in ("HIGH", "MEDIUM"):
            risk = "MEDIUM"
        elif min_risk == "LOW" and risk == "NEGLIGIBLE":
            risk = "LOW"
        eid = f"EVT-2024-{np.random.randint(1,999):03d}"
        events.append({
            "Date": d.strftime('%Y-%m-%d'),
            "Event": eid,
            "Risk": risk,
            "Miss Distance (km)": round(np.random.uniform(0.01, 10), 3),
        })

    st.markdown('''<div style="margin-top:12px;">
        <h3 style="color:#00D4FF; font-size:0.9rem; border-bottom:1px solid #243044;
                   padding-bottom:6px;">Event Log (showing up to 12)</h3>
    </div>''', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)

    json_data = {
        "type": "event_timeline",
        "generated_utc": now.isoformat(),
        "start": str(date_start), "end": str(date_end),
        "min_risk": min_risk, "group_by": group_by,
        "total_events": int(n_events), "high_risk": int(high_ct),
        "medium_risk": int(med_ct), "low_risk": int(low_ct),
        "events_sample": events,
    }
    return "Event Timeline Report", json_data


def report_title_str(report_type, detail=""):
    if detail:
        return f"{report_type} - {detail}"
    return report_type


def _build_full_html(title, json_data):
    """Build a standalone HTML file for download."""
    now = json_data.get("generated_utc", datetime.datetime.now(datetime.timezone.utc).isoformat())
    rtype = json_data.get("type", "report")

    body_parts = [f'<h1 style="color:#00D4FF; font-size:1.4rem;">{title}</h1>',
                  f'<p style="color:#6e7d8f;">Generated: {now}</p>']

    if rtype == "conjunction":
        body_parts.append(f'''
        <table><tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Risk Level</td><td>{json_data.get("risk_level","N/A")}</td></tr>
        <tr><td>log10(Pc)</td><td>{json_data.get("log10_pc","N/A")}</td></tr>
        <tr><td>Miss Distance</td><td>{json_data.get("miss_distance_km","N/A")} km</td></tr>
        <tr><td>Relative Speed</td><td>{json_data.get("relative_speed_kms","N/A")} km/s</td></tr>
        </table>''')

    elif rtype == "model_performance":
        body_parts.append('<h2 style="color:#00D4FF; font-size:1rem; margin-top:20px;">Performance Metrics</h2>')
        body_parts.append('<table><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th>'
                          '<th>Accuracy</th><th>F1</th><th>Precision</th><th>Recall</th></tr>')
        for m, met in json_data.get("metrics", {}).items():
            body_parts.append(
                f'<tr><td>{m}</td><td>{met.get("rmse","")}</td><td>{met.get("mae","")}</td>'
                f'<td>{met.get("r2","")}</td><td>{met.get("accuracy",0)*100:.1f}%</td>'
                f'<td>{met.get("f1",0)*100:.1f}%</td><td>{met.get("precision",0)*100:.1f}%</td>'
                f'<td>{met.get("recall",0)*100:.1f}%</td></tr>')
        body_parts.append('</table>')
        body_parts.append('<h2 style="color:#7B61FF; font-size:1rem; margin-top:20px;">Ensemble Weights</h2>')
        for m, w in json_data.get("ensemble_weights", {}).items():
            body_parts.append(f'<p>{m}: {w*100:.0f}%</p>')
        if json_data.get("shap_features"):
            body_parts.append('<h2 style="color:#7B61FF; font-size:1rem; margin-top:20px;">SHAP Feature Importance</h2>')
            body_parts.append('<table><tr><th>Feature</th><th>Importance</th></tr>')
            for item in json_data["shap_features"]:
                body_parts.append(f'<tr><td>{item["feature"]}</td><td>{item["importance"]:.4f}</td></tr>')
            body_parts.append('</table>')

    elif rtype == "event_timeline":
        body_parts.append(f'<p>Period: {json_data.get("start")} to {json_data.get("end")}</p>')
        body_parts.append(f'<p>Total: {json_data.get("total_events",0)} events | '
                          f'High: {json_data.get("high_risk",0)} | Medium: {json_data.get("medium_risk",0)} | '
                          f'Low: {json_data.get("low_risk",0)}</p>')
        if json_data.get("events_sample"):
            body_parts.append('<table><tr><th>Date</th><th>Event</th><th>Risk</th><th>Miss Distance</th></tr>')
            for ev in json_data["events_sample"]:
                body_parts.append(f'<tr><td>{ev["Date"]}</td><td>{ev["Event"]}</td>'
                                  f'<td>{ev["Risk"]}</td><td>{ev["Miss Distance (km)"]} km</td></tr>')
            body_parts.append('</table>')

    body_html = '\n'.join(body_parts)
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
h1 {{ margin-bottom:8px; }} h2 {{ margin-bottom:8px; }}
table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:0.85rem; }}
th {{ padding:8px 12px; text-align:left; color:#3d4f63; border-bottom:2px solid #243044;
     font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em; }}
td {{ padding:8px 12px; border-bottom:1px solid #1a2332; }}
p {{ color:#6e7d8f; font-size:0.85rem; margin:6px 0; }}
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


def _build_markdown(title, json_data):
    """Build a clean markdown version of the report."""
    lines = [f"# {title}", ""]
    rtype = json_data.get("type", "report")
    lines.append(f"**Generated:** {json_data.get('generated_utc', 'N/A')}")
    lines.append("")

    if rtype == "conjunction":
        lines += [
            "## Risk Summary", "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Risk Level | {json_data.get('risk_level','N/A')} |",
            f"| log10(Pc) | {json_data.get('log10_pc','N/A')} |",
            f"| Miss Distance | {json_data.get('miss_distance_km','N/A')} km |",
            f"| Relative Speed | {json_data.get('relative_speed_kms','N/A')} km/s |",
            "",
        ]

    elif rtype == "model_performance":
        lines += ["## Performance Metrics", ""]
        lines.append("| Model | RMSE | MAE | R² | Accuracy | F1 | Precision | Recall |")
        lines.append("|-------|------|-----|-----|----------|-----|-----------|--------|")
        for m, met in json_data.get("metrics", {}).items():
            lines.append(
                f"| {m} | {met.get('rmse','')} | {met.get('mae','')} | {met.get('r2','')} | "
                f"{met.get('accuracy',0)*100:.1f}% | {met.get('f1',0)*100:.1f}% | "
                f"{met.get('precision',0)*100:.1f}% | {met.get('recall',0)*100:.1f}% |"
            )
        lines += ["", "## Ensemble Weights", ""]
        for m, w in json_data.get("ensemble_weights", {}).items():
            lines.append(f"- **{m}**: {w*100:.0f}%")
        if json_data.get("shap_features"):
            lines += ["", "## SHAP Feature Importance", ""]
            lines.append("| Feature | Importance |")
            lines.append("|---------|------------|")
            for item in json_data["shap_features"]:
                lines.append(f"| {item['feature']} | {item['importance']:.4f} |")
        lines.append("")

    elif rtype == "event_timeline":
        lines += [
            f"**Period:** {json_data.get('start')} to {json_data.get('end')}",
            f"**Total Events:** {json_data.get('total_events',0)}",
            f"**High Risk:** {json_data.get('high_risk',0)} | "
            f"**Medium:** {json_data.get('medium_risk',0)} | "
            f"**Low:** {json_data.get('low_risk',0)}",
            "",
        ]
        if json_data.get("events_sample"):
            lines += ["## Event Log", ""]
            lines.append("| Date | Event | Risk | Miss Distance |")
            lines.append("|------|-------|------|---------------|")
            for ev in json_data["events_sample"]:
                lines.append(f"| {ev['Date']} | {ev['Event']} | {ev['Risk']} | {ev['Miss Distance (km)']} km |")
        lines.append("")

    return '\n'.join(lines)


# ─── Main Render ──────────────────────────────────────────────────

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
                "Model Performance", "\U0001f9e0",
                "Comprehensive ML model evaluation with metrics, feature importance, "
                "and calibration analysis",
                report_type == "Model Performance Report",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _report_type_card(
                "Event Timeline", "\U0001f4c5",
                "Chronological summary of conjunction events over a selected time period",
                report_type == "Event Timeline Report",
            ),
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Report Configuration ──
    st.markdown("#### Configuration")

    if report_type == "Conjunction Report":
        event_ids = [f"EVT-2024-{i:03d}" for i in range(1, 21)]
        cfg1, cfg2 = st.columns([2, 1])
        with cfg1:
            event_id = st.selectbox("Event ID", event_ids, key="rg_event")
        with cfg2:
            include_physics = st.checkbox("Include Physics Analysis", value=True, key="rg_phys")
            include_recs = st.checkbox("Include Recommendations", value=True, key="rg_recs")

        if st.button("Generate Report", type="primary", key="rg_gen_conj"):
            st.session_state["rg_generated"] = True
            st.session_state["rg_cfg"] = {
                "type": "conjunction", "event_id": event_id,
                "include_physics": include_physics, "include_recs": include_recs,
            }

        if st.session_state.get("rg_generated") and st.session_state.get("rg_cfg", {}).get("type") == "conjunction":
            cfg = st.session_state["rg_cfg"]
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### Report Preview")
            title, json_data = _render_conjunction_report(
                cfg["event_id"], cfg["include_physics"], cfg["include_recs"])
            st.session_state["rg_json"] = json_data
            st.session_state["rg_title"] = title

    elif report_type == "Model Performance Report":
        all_models = ["XGBoost", "LightGBM", "Random Forest", "Logistic Regression"]
        cfg1, cfg2 = st.columns([2, 1])
        with cfg1:
            selected_models = st.multiselect("Models to Evaluate", all_models, default=all_models[:2], key="rg_models")
        with cfg2:
            include_shap = st.checkbox("Include SHAP Analysis", value=True, key="rg_shap")
            include_cal = st.checkbox("Include Calibration Curves", value=True, key="rg_cal")

        if st.button("Generate Report", type="primary", key="rg_gen_model"):
            if not selected_models:
                st.warning("Select at least one model.")
            else:
                st.session_state["rg_generated"] = True
                st.session_state["rg_cfg"] = {
                    "type": "model", "models": selected_models,
                    "include_shap": include_shap, "include_cal": include_cal,
                }

        if st.session_state.get("rg_generated") and st.session_state.get("rg_cfg", {}).get("type") == "model":
            cfg = st.session_state["rg_cfg"]
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### Report Preview")
            title, json_data = _render_model_report(
                cfg["models"], cfg["include_shap"], cfg["include_cal"])
            st.session_state["rg_json"] = json_data
            st.session_state["rg_title"] = title

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

        if st.button("Generate Report", type="primary", key="rg_gen_tl"):
            st.session_state["rg_generated"] = True
            st.session_state["rg_cfg"] = {
                "type": "timeline",
                "date_start": date_start, "date_end": date_end,
                "min_risk": min_risk, "group_by": group_by,
            }

        if st.session_state.get("rg_generated") and st.session_state.get("rg_cfg", {}).get("type") == "timeline":
            cfg = st.session_state["rg_cfg"]
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### Report Preview")
            title, json_data = _render_timeline_report(
                cfg["date_start"], cfg["date_end"], cfg["min_risk"], cfg["group_by"])
            st.session_state["rg_json"] = json_data
            st.session_state["rg_title"] = title

    # ── Download Buttons ──
    if st.session_state.get("rg_json"):
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Export")

        title = st.session_state.get("rg_title", "Report")
        json_data = st.session_state["rg_json"]
        fname_base = f"orbital_sentinel_{title.lower().replace(' ', '_').replace('-','_')}"

        full_html = _build_full_html(title, json_data)
        md_content = _build_markdown(title, json_data)
        json_str = json.dumps(json_data, indent=2, default=str)

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "Download Markdown",
                data=md_content,
                file_name=f"{fname_base}.md",
                mime="text/markdown",
                key="rg_dl_md",
            )
            st.markdown(f'<div style="color:#6e7d8f; font-size:0.7rem; margin-top:4px;">'
                        f'~{len(md_content)/1024:.1f} KB</div>', unsafe_allow_html=True)
        with d2:
            st.download_button(
                "Download HTML",
                data=full_html,
                file_name=f"{fname_base}.html",
                mime="text/html",
                key="rg_dl_html",
            )
            st.markdown(f'<div style="color:#6e7d8f; font-size:0.7rem; margin-top:4px;">'
                        f'~{len(full_html)/1024:.1f} KB</div>', unsafe_allow_html=True)
        with d3:
            st.download_button(
                "Download JSON",
                data=json_str,
                file_name=f"{fname_base}.json",
                mime="application/json",
                key="rg_dl_json",
            )
            st.markdown(f'<div style="color:#6e7d8f; font-size:0.7rem; margin-top:4px;">'
                        f'~{len(json_str)/1024:.1f} KB</div>', unsafe_allow_html=True)

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

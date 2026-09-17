"""Orbital Sentinel — Professional Satellite Conjunction Risk Dashboard."""

import sys
import json
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODELS_DIR = PROJECT_ROOT / "models_saved"
PROOFS_DIR = PROJECT_ROOT / "proofs"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "esa_kelvins"

# ---------------------------------------------------------------------------
# Page config & Theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Orbital Sentinel",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0E1117;
    --bg-secondary: #161B22;
    --bg-card: #1C2333;
    --bg-card-hover: #222D3F;
    --border: #30363D;
    --text-primary: #E6EDF3;
    --text-secondary: #8B949E;
    --accent-cyan: #00D4FF;
    --accent-green: #00FF88;
    --accent-amber: #FFB800;
    --accent-red: #FF3366;
    --accent-purple: #7C5CFC;
}

.main .block-container { padding-top: 1.5rem; max-width: 1400px; }

h1, h2, h3, h4 { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; }

.metric-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}
.metric-card:hover {
    border-color: var(--accent-cyan);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0,212,255,0.1);
}
.metric-card .metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-bottom: 4px;
}
.metric-card .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
    line-height: 1.2;
}
.metric-card .metric-value.cyan { color: var(--accent-cyan); }
.metric-card .metric-value.green { color: var(--accent-green); }
.metric-card .metric-value.amber { color: var(--accent-amber); }
.metric-card .metric-value.red { color: var(--accent-red); }
.metric-card .metric-value.purple { color: var(--accent-purple); }

.risk-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.risk-badge.high { background: rgba(255,51,102,0.15); color: #FF3366; border: 1px solid rgba(255,51,102,0.3); }
.risk-badge.medium { background: rgba(255,184,0,0.15); color: #FFB800; border: 1px solid rgba(255,184,0,0.3); }
.risk-badge.low { background: rgba(0,212,255,0.15); color: #00D4FF; border: 1px solid rgba(0,212,255,0.3); }
.risk-badge.negligible { background: rgba(0,255,136,0.15); color: #00FF88; border: 1px solid rgba(0,255,136,0.3); }

.alert-card {
    background: var(--bg-card);
    border-left: 4px solid;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-family: 'Inter', sans-serif;
}
.alert-card.critical { border-left-color: #FF3366; }
.alert-card.warning { border-left-color: #FFB800; }
.alert-card.info { border-left-color: #00D4FF; }

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
.status-dot.active { background: #00FF88; }
.status-dot.warning { background: #FFB800; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 24px 0;
}

div[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
}

div[data-testid="stSidebar"] .stRadio > label { font-weight: 500; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
}
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, color: str = ""):
    color_class = f" {color}" if color else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value{color_class}">{value}</div>
    </div>
    """


def risk_badge(category: str) -> str:
    cat_lower = category.lower() if category else "low"
    if "high" in cat_lower:
        return '<span class="risk-badge high">HIGH RISK</span>'
    if "med" in cat_lower:
        return '<span class="risk-badge medium">MEDIUM RISK</span>'
    if "low" in cat_lower:
        return '<span class="risk-badge low">LOW RISK</span>'
    return '<span class="risk-badge negligible">NEGLIGIBLE</span>'


def alert_card(title: str, detail: str, level: str = "info") -> str:
    return f"""
    <div class="alert-card {level}">
        <strong>{title}</strong><br>
        <span style="color: #8B949E; font-size: 0.85rem;">{detail}</span>
    </div>
    """


def risk_color(val: float) -> str:
    if val > -5: return "#FF3366"
    if val > -7: return "#FFB800"
    if val > -15: return "#00D4FF"
    return "#00FF88"


def risk_label(val: float) -> str:
    if val > -5: return "HIGH"
    if val > -7: return "MEDIUM"
    if val > -15: return "LOW"
    return "NEGLIGIBLE"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dataset():
    path = DATA_DIR / "train_data.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_pipeline_results():
    path = PROOFS_DIR / "pipeline_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_feature_names():
    path = MODELS_DIR / "feature_names.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    path = MODELS_DIR / f"{model_name}_model.joblib"
    return joblib.load(str(path)) if path.exists() else None


@st.cache_resource(show_spinner=False)
def load_scaler():
    path = MODELS_DIR / "scaler.joblib"
    return joblib.load(str(path)) if path.exists() else None


def available_models():
    if not MODELS_DIR.exists():
        return []
    return sorted(p.stem.replace("_model", "") for p in MODELS_DIR.glob("*_model.joblib"))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 16px;">
        <div style="font-size:2.5rem;">&#127760;</div>
        <div style="font-size:1.4rem; font-weight:700; color:#E6EDF3; letter-spacing:-0.02em;">
            ORBITAL SENTINEL
        </div>
        <div style="font-size:0.75rem; color:#8B949E; letter-spacing:0.08em; text-transform:uppercase;">
            Conjunction Risk Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        [
            "Mission Control",
            "Risk Assessment",
            "Event Timeline",
            "Model Performance",
            "Feature Importance",
            "Physics Lab",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 0.7rem; color: #8B949E; text-align: center; padding: 8px 0;">
        <div><span class="status-dot active"></span>System Online</div>
        <div style="margin-top:4px;">v0.1.0 &middot; XGBoost Engine</div>
    </div>
    """, unsafe_allow_html=True)


# ===================================================================
# Page 1 — Mission Control (Overview Dashboard)
# ===================================================================

if page == "Mission Control":
    st.markdown("## Mission Control")
    st.markdown('<p style="color:#8B949E; margin-top:-10px;">Real-time conjunction risk overview</p>',
                unsafe_allow_html=True)

    df = load_dataset()
    results = load_pipeline_results()

    if df is None:
        st.warning("Dataset not found. Run the ingestion pipeline first.")
        st.stop()

    # Hero metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(metric_card("Total Conjunctions", f"{len(df):,}", "cyan"), unsafe_allow_html=True)
    with c2:
        n_events = df["event_id"].nunique() if "event_id" in df.columns else 0
        st.markdown(metric_card("Unique Events", f"{n_events:,}", "purple"), unsafe_allow_html=True)
    with c3:
        if "risk" in df.columns:
            high_risk = (df["risk"] > -5).sum()
            st.markdown(metric_card("High Risk", f"{high_risk:,}", "red"), unsafe_allow_html=True)
        else:
            st.markdown(metric_card("High Risk", "N/A"), unsafe_allow_html=True)
    with c4:
        if results:
            best_rmse = results.get("evaluation", {}).get("overall", {}).get("rmse", "N/A")
            st.markdown(metric_card("Model RMSE", f"{best_rmse:.3f}" if isinstance(best_rmse, float) else "N/A", "green"),
                        unsafe_allow_html=True)
        else:
            st.markdown(metric_card("Model RMSE", "N/A"), unsafe_allow_html=True)
    with c5:
        if results:
            r2 = results.get("evaluation", {}).get("overall", {}).get("r2", "N/A")
            st.markdown(metric_card("Model R²", f"{r2:.4f}" if isinstance(r2, float) else "N/A", "green"),
                        unsafe_allow_html=True)
        else:
            st.markdown(metric_card("Model R²", "N/A"), unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Two column layout: risk distribution + alert feed
    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### Risk Distribution")
        if "risk" in df.columns:
            fig = go.Figure()

            bins = np.histogram_bin_edges(df["risk"].dropna(), bins=80)
            counts, edges = np.histogram(df["risk"].dropna(), bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2
            colors_list = [risk_color(c) for c in centers]

            fig.add_trace(go.Bar(
                x=centers, y=counts,
                marker_color=colors_list,
                opacity=0.85,
                hovertemplate="log10(Pc): %{x:.1f}<br>Count: %{y:,}<extra></extra>",
            ))

            fig.add_vline(x=-5, line_dash="dash", line_color="#FF3366", line_width=2,
                          annotation_text="HIGH", annotation_font_color="#FF3366")
            fig.add_vline(x=-7, line_dash="dash", line_color="#FFB800", line_width=2,
                          annotation_text="MEDIUM", annotation_font_color="#FFB800")

            fig.update_layout(
                xaxis_title="log10(Collision Probability)",
                yaxis_title="Count",
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                font=dict(color="#E6EDF3", family="Inter"),
                xaxis=dict(gridcolor="#1a2332"),
                yaxis=dict(gridcolor="#1a2332"),
                margin=dict(l=50, r=20, t=10, b=50),
                height=380,
                bargap=0.02,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Alert Feed")

        if "risk" in df.columns:
            high_events = df[df["risk"] > -5].head(8)
            if len(high_events) > 0:
                for _, row in high_events.iterrows():
                    eid = row.get("event_id", "?")
                    risk_val = row.get("risk", -30)
                    miss = row.get("miss_distance", 0)
                    lvl = "critical" if risk_val > -3 else "warning"
                    st.markdown(
                        alert_card(
                            f"Event {eid}",
                            f"Risk: {risk_val:.1f} | Miss: {miss:,.0f}m | {risk_label(risk_val)}",
                            lvl,
                        ),
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(alert_card("All Clear", "No high-risk conjunctions detected", "info"),
                            unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Risk category breakdown
    st.markdown("#### Risk Category Breakdown")
    if "risk" in df.columns:
        cat_data = {
            "NEGLIGIBLE (< -15)": int((df["risk"] <= -15).sum()),
            "LOW (-15 to -7)": int(((df["risk"] > -15) & (df["risk"] <= -7)).sum()),
            "MEDIUM (-7 to -5)": int(((df["risk"] > -7) & (df["risk"] <= -5)).sum()),
            "HIGH (> -5)": int((df["risk"] > -5).sum()),
        }
        cat_colors = ["#00FF88", "#00D4FF", "#FFB800", "#FF3366"]

        cc1, cc2 = st.columns([1, 2])
        with cc1:
            fig_pie = go.Figure(go.Pie(
                labels=list(cat_data.keys()),
                values=list(cat_data.values()),
                marker=dict(colors=cat_colors, line=dict(color="#0E1117", width=2)),
                textinfo="percent+label",
                textfont=dict(size=11),
                hole=0.55,
            ))
            fig_pie.update_layout(
                paper_bgcolor="#0E1117",
                font=dict(color="#E6EDF3", family="Inter"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=320,
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with cc2:
            for cat_name, count in cat_data.items():
                pct = count / len(df) * 100
                color = cat_colors[list(cat_data.keys()).index(cat_name)]
                st.markdown(f"""
                <div style="display:flex; align-items:center; margin-bottom:8px;">
                    <div style="width:12px; height:12px; border-radius:3px; background:{color}; margin-right:10px;"></div>
                    <div style="flex:1;">
                        <span style="color:#E6EDF3; font-weight:500;">{cat_name}</span><br>
                        <span style="color:#8B949E; font-size:0.85rem;">{count:,} events ({pct:.1f}%)</span>
                    </div>
                    <div style="font-family:'JetBrains Mono'; font-size:1.2rem; color:{color}; font-weight:600;">
                        {count:,}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Model comparison mini-table
    if results and "model_metrics" in results:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Model Leaderboard")
        metrics = results["model_metrics"]
        rows = []
        for name, data in metrics.items():
            val = data.get("val", {})
            rows.append({
                "Model": name,
                "MAE": val.get("mae"),
                "RMSE": val.get("rmse"),
                "R²": val.get("r2"),
                "Correlation": val.get("correlation"),
            })
        leaderboard = pd.DataFrame(rows).set_index("Model").sort_values("RMSE")
        st.dataframe(
            leaderboard.style.format("{:.4f}", na_rep="-").highlight_min(
                subset=["MAE", "RMSE"], color="rgba(0,255,136,0.15)"
            ).highlight_max(
                subset=["R²", "Correlation"], color="rgba(0,212,255,0.15)"
            ),
            use_container_width=True,
        )


# ===================================================================
# Page 2 — Risk Assessment
# ===================================================================

elif page == "Risk Assessment":
    st.markdown("## Risk Assessment")
    st.markdown('<p style="color:#8B949E; margin-top:-10px;">Full conjunction analysis with ML + Physics fusion</p>',
                unsafe_allow_html=True)

    models_available = available_models()
    feature_names = load_feature_names()
    scaler = load_scaler()
    df = load_dataset()

    if not models_available or feature_names is None or df is None:
        st.warning("Trained model, feature names, and dataset are required. Run the pipeline first.")
        st.stop()

    ctl1, ctl2 = st.columns([1, 2])
    with ctl1:
        selected_model = st.selectbox("Model", models_available)
    with ctl2:
        event_ids = sorted(df["event_id"].unique()) if "event_id" in df.columns else list(range(len(df)))
        selected_event = st.selectbox("Conjunction Event", event_ids)

    model = load_model(selected_model)
    if model is None:
        st.error(f"Failed to load model: {selected_model}")
        st.stop()

    event_rows = df[df["event_id"] == selected_event] if "event_id" in df.columns else df.iloc[[selected_event]]
    if event_rows.empty:
        st.error("No data for selected event.")
        st.stop()

    row = event_rows.iloc[-1]
    row_dict = row.to_dict()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Run inference
    with st.spinner("Running full analysis..."):
        try:
            from orbital_sentinel.inference.pipeline import predict_single_event
            result = predict_single_event(row_dict, model, scaler, feature_names)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # Results layout
    st.markdown("#### Prediction Results")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(metric_card("Predicted Risk", f"{result['prediction']:.2f}",
                                "red" if result["prediction"] > -5 else "cyan"), unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Category</div>
            <div style="padding-top:4px;">{risk_badge(result['risk_category'])}</div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        conf = result.get("confidence", 0)
        st.markdown(metric_card("Confidence", f"{conf:.2f}" if conf else "N/A", "green"), unsafe_allow_html=True)
    with r4:
        interval = result.get("interval")
        iv_text = f"[{interval[0]:.1f}, {interval[1]:.1f}]" if interval else "N/A"
        st.markdown(metric_card("90% Interval", iv_text, "purple"), unsafe_allow_html=True)

    # Event details
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(metric_card("Event ID", str(selected_event)), unsafe_allow_html=True)
    with d2:
        tca = row.get("time_to_tca")
        st.markdown(metric_card("Time to TCA", f"{tca:.3f} days" if pd.notna(tca) else "N/A"), unsafe_allow_html=True)
    with d3:
        miss = row.get("miss_distance", 0)
        st.markdown(metric_card("Miss Distance", f"{miss:,.0f} m"), unsafe_allow_html=True)
    with d4:
        true_risk = row.get("risk")
        st.markdown(metric_card("True Risk", f"{true_risk:.2f}" if pd.notna(true_risk) else "N/A"), unsafe_allow_html=True)

    # Physics verification
    physics = result.get("physics_result")
    if physics:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Physics Verification")

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            log_pc = physics.get("log10_pc", float("nan"))
            st.markdown(metric_card("Analytic Pc",
                                    f"10^{log_pc:.2f}" if np.isfinite(log_pc) else "N/A", "cyan"),
                        unsafe_allow_html=True)
        with p2:
            st.markdown(metric_card("Covariance",
                                    "VALID" if physics.get("covariance_valid") else "INVALID",
                                    "green" if physics.get("covariance_valid") else "red"),
                        unsafe_allow_html=True)
        with p3:
            st.markdown(metric_card("Short Encounter",
                                    "YES" if physics.get("short_encounter_valid") else "NO",
                                    "green" if physics.get("short_encounter_valid") else "amber"),
                        unsafe_allow_html=True)
        with p4:
            angle = physics.get("geometry", {}).get("approach_angle_deg", 0)
            st.markdown(metric_card("Approach Angle", f"{angle:.1f} deg"), unsafe_allow_html=True)

    # Fusion
    fusion = result.get("fusion")
    if fusion:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Fused Risk Assessment")

        f1, f2, f3 = st.columns(3)
        fused_risk = fusion.get("fused_risk", fusion.get("risk_estimate", result["prediction"]))
        fused_cat = fusion.get("risk_category", result["risk_category"])
        fused_conf = fusion.get("confidence", result.get("confidence", 0.5))

        with f1:
            st.markdown(metric_card("Fused Risk", f"{fused_risk:.2f}",
                                    "red" if fused_risk > -5 else "cyan"), unsafe_allow_html=True)
        with f2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Fused Category</div>
                <div style="padding-top:4px;">{risk_badge(fused_cat)}</div>
            </div>
            """, unsafe_allow_html=True)
        with f3:
            st.markdown(metric_card("Fused Confidence", f"{fused_conf:.2f}", "green"), unsafe_allow_html=True)

    # Explanation
    explanation = result.get("explanation")
    if explanation:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### AI Explanation")
        st.success(explanation)

    # Top features
    top_feats = result.get("top_features")
    if top_feats:
        st.markdown("#### Top Risk Drivers")
        tf_df = pd.DataFrame(top_feats)
        if "shap_value" in tf_df.columns:
            colors_bar = ["#FF3366" if v > 0 else "#00D4FF" for v in tf_df["shap_value"]]
            fig_shap = go.Figure(go.Bar(
                x=tf_df["shap_value"], y=tf_df["feature"],
                orientation="h", marker_color=colors_bar,
            ))
            fig_shap.update_layout(
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                font=dict(color="#E6EDF3", family="Inter"),
                xaxis=dict(title="SHAP Value", gridcolor="#1a2332"),
                yaxis_gridcolor="#1a2332",
                margin=dict(l=200, r=20, t=10, b=40),
                height=300,
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("Red = increases risk | Blue = decreases risk")


# ===================================================================
# Page 4 — Event Timeline
# ===================================================================

elif page == "Event Timeline":
    st.markdown("## Event Timeline")
    st.markdown('<p style="color:#8B949E; margin-top:-10px;">Risk evolution as TCA approaches</p>',
                unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.warning("Dataset not found.")
        st.stop()

    from orbital_sentinel.events.sequencing import compute_risk_trend, identify_escalating_events

    # Find events with multiple CDMs
    if "event_id" in df.columns:
        event_counts = df["event_id"].value_counts()
        multi_cdm_events = event_counts[event_counts > 1].index.tolist()

        if multi_cdm_events:
            ctl1, ctl2 = st.columns([2, 1])
            with ctl1:
                selected_event = st.selectbox(
                    "Select Event (multi-CDM events only)",
                    sorted(multi_cdm_events)[:200],
                )
            with ctl2:
                st.markdown(metric_card("Multi-CDM Events", f"{len(multi_cdm_events):,}", "cyan"),
                            unsafe_allow_html=True)

            event_rows = df[df["event_id"] == selected_event].sort_values("time_to_tca", ascending=False)
            sequence = event_rows.to_dict("records")

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

            # Risk timeline chart
            st.markdown("#### Risk Evolution")
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(
                x=event_rows["time_to_tca"], y=event_rows["risk"],
                mode="lines+markers", name="log10(Pc)",
                line=dict(color="#FF3366", width=2), marker=dict(size=7),
            ))
            if "miss_distance" in event_rows.columns:
                fig_timeline.add_trace(go.Scatter(
                    x=event_rows["time_to_tca"], y=event_rows["miss_distance"],
                    mode="lines+markers", name="Miss Distance (m)",
                    line=dict(color="#00D4FF", width=1, dash="dot"), marker=dict(size=4),
                    yaxis="y2",
                ))
                fig_timeline.update_layout(yaxis2=dict(title="Miss Distance (m)", overlaying="y", side="right"))
            fig_timeline.update_layout(
                xaxis_title="Time to TCA (days)", yaxis_title="log10(Pc)",
                xaxis=dict(autorange="reversed"), height=350,
                margin=dict(t=20, b=30),
                template="plotly_dark",
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

            # Trend analysis
            trend = compute_risk_trend(sequence)

            t1, t2, t3, t4 = st.columns(4)
            with t1:
                trend_text = trend.get("trend", "unknown").upper()
                trend_color = "red" if trend_text == "INCREASING" else ("green" if trend_text == "DECREASING" else "amber")
                st.markdown(metric_card("Trend", trend_text, trend_color), unsafe_allow_html=True)
            with t2:
                st.markdown(metric_card("CDM Count", str(trend.get("n_points", 0)), "cyan"), unsafe_allow_html=True)
            with t3:
                change = trend.get("risk_change", 0)
                st.markdown(metric_card("Risk Change", f"{change:+.2f}", "red" if change > 0 else "green"),
                            unsafe_allow_html=True)
            with t4:
                conv = "YES" if trend.get("is_converging") else "NO"
                st.markdown(metric_card("Converging", conv, "green" if conv == "YES" else "amber"),
                            unsafe_allow_html=True)

            # CDM sequence table
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### CDM Sequence")
            display_cols = ["time_to_tca", "risk", "miss_distance", "relative_speed"]
            avail_cols = [c for c in display_cols if c in event_rows.columns]
            st.dataframe(
                event_rows[avail_cols].style.format({
                    "time_to_tca": "{:.3f}",
                    "risk": "{:.2f}",
                    "miss_distance": "{:,.0f}",
                    "relative_speed": "{:,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info("No events with multiple CDMs found in the dataset.")

        # Escalating events
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Escalating Events")
        st.markdown('<p style="color:#8B949E;">Events where risk is increasing and exceeds threshold</p>',
                    unsafe_allow_html=True)

        with st.spinner("Analyzing event trends..."):
            escalating = identify_escalating_events(df, threshold=-7.0)

        if escalating:
            for ev in escalating[:10]:
                eid = ev["event_id"]
                latest = ev["latest_risk"]
                n_cdms = ev["n_cdms"]
                lvl = "critical" if latest > -5 else "warning"
                st.markdown(
                    alert_card(
                        f"Event {eid} -- ESCALATING",
                        f"Latest risk: {latest:.2f} | CDMs: {n_cdms} | Trend: {ev['trend']['trend']}",
                        lvl,
                    ),
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(alert_card("All Clear", "No escalating events detected above threshold", "info"),
                        unsafe_allow_html=True)

        # Risk matrix
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Risk Matrix -- Miss Distance vs Risk")
        sample_summary = df.groupby("event_id").last().reset_index().head(500)
        if "miss_distance" in sample_summary.columns and "risk" in sample_summary.columns:
            fig_matrix = px.scatter(
                sample_summary, x="miss_distance", y="risk",
                color="risk", color_continuous_scale="RdYlGn_r",
                labels={"miss_distance": "Miss Distance (m)", "risk": "log10(Pc)"},
                opacity=0.6,
            )
            fig_matrix.add_hline(y=-5.0, line_dash="dash", line_color="#FF3366",
                                 annotation_text="High-risk threshold")
            fig_matrix.update_layout(height=400, margin=dict(t=20, b=30), template="plotly_dark")
            st.plotly_chart(fig_matrix, use_container_width=True)


# ===================================================================
# Page 5 — Model Performance
# ===================================================================

elif page == "Model Performance":
    st.markdown("## Model Performance")
    st.markdown('<p style="color:#8B949E; margin-top:-10px;">Training results and model comparison</p>',
                unsafe_allow_html=True)

    results = load_pipeline_results()
    if results is None:
        st.warning("Pipeline results not found. Run the training pipeline first.")
        st.stop()

    model_metrics = results.get("model_metrics", {})
    if not model_metrics:
        st.info("No model metrics found.")
        st.stop()

    # Model comparison cards
    st.markdown("#### Model Comparison")
    cols = st.columns(len(model_metrics))
    best_name = min(model_metrics, key=lambda n: model_metrics[n].get("val", {}).get("rmse", 999))

    for col, (name, data) in zip(cols, model_metrics.items()):
        val = data.get("val", {})
        is_best = name == best_name
        with col:
            border_color = "#00FF88" if is_best else "#30363D"
            badge = '<span style="color:#00FF88; font-size:0.7rem; font-weight:600;">BEST</span>' if is_best else ""
            st.markdown(f"""
            <div style="background:#1C2333; border:2px solid {border_color}; border-radius:12px; padding:20px; text-align:center;">
                <div style="font-size:1.1rem; font-weight:600; color:#E6EDF3;">{name} {badge}</div>
                <div style="margin-top:12px;">
                    <div style="color:#8B949E; font-size:0.7rem; text-transform:uppercase;">RMSE</div>
                    <div style="font-family:'JetBrains Mono'; font-size:1.5rem; color:{'#00FF88' if is_best else '#00D4FF'};">
                        {val.get('rmse', 0):.3f}
                    </div>
                </div>
                <div style="margin-top:8px;">
                    <div style="color:#8B949E; font-size:0.7rem; text-transform:uppercase;">R-SQUARED</div>
                    <div style="font-family:'JetBrains Mono'; font-size:1.2rem; color:#E6EDF3;">
                        {val.get('r2', 0):.4f}
                    </div>
                </div>
                <div style="margin-top:8px;">
                    <div style="color:#8B949E; font-size:0.7rem; text-transform:uppercase;">MAE</div>
                    <div style="font-family:'JetBrains Mono'; font-size:1.2rem; color:#E6EDF3;">
                        {val.get('mae', 0):.4f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Bar chart comparison
    st.markdown("#### Metric Comparison")
    metrics_to_compare = ["rmse", "mae", "r2", "correlation"]
    tabs = st.tabs([m.upper() for m in metrics_to_compare])

    for tab, metric_key in zip(tabs, metrics_to_compare):
        with tab:
            names = list(model_metrics.keys())
            values = [model_metrics[n].get("val", {}).get(metric_key, 0) for n in names]
            colors = ["#00FF88" if n == best_name else "#00D4FF" for n in names]

            fig = go.Figure(go.Bar(
                x=names, y=values,
                marker_color=colors,
                text=[f"{v:.4f}" for v in values],
                textposition="auto",
                textfont=dict(color="white", size=14),
            ))
            fig.update_layout(
                yaxis_title=metric_key.upper(),
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                font=dict(color="#E6EDF3", family="Inter"),
                xaxis=dict(gridcolor="#1a2332"),
                yaxis=dict(gridcolor="#1a2332"),
                margin=dict(l=50, r=20, t=10, b=50),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Train vs Val comparison
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### Train vs Validation Performance")

    overfit_data = []
    for name, data in model_metrics.items():
        train = data.get("train", {})
        val = data.get("val", {})
        overfit_data.append({
            "Model": name,
            "Train RMSE": train.get("rmse", 0),
            "Val RMSE": val.get("rmse", 0),
            "Gap": val.get("rmse", 0) - train.get("rmse", 0),
            "Train R²": train.get("r2", 0),
            "Val R²": val.get("r2", 0),
        })
    overfit_df = pd.DataFrame(overfit_data).set_index("Model")
    st.dataframe(
        overfit_df.style.format("{:.4f}").background_gradient(subset=["Gap"], cmap="RdYlGn_r"),
        use_container_width=True,
    )

    # Per-risk-band evaluation
    evaluation = results.get("evaluation", {})
    by_band = evaluation.get("by_band", {})
    if by_band:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Per-Risk-Band Performance (Best Model)")
        band_rows = []
        for band_name, bm in by_band.items():
            band_rows.append({
                "Band": band_name.upper(),
                "Count": bm.get("n", bm.get("count", 0)),
                "MAE": bm.get("mae", 0),
                "RMSE": bm.get("rmse", 0),
                "Correlation": bm.get("correlation", 0),
            })
        band_df = pd.DataFrame(band_rows).set_index("Band")
        st.dataframe(
            band_df.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "Correlation": "{:.4f}", "Count": "{:,}"}),
            use_container_width=True,
        )

    # Binary classification metrics
    thresh = evaluation.get("threshold_-5", {})
    if thresh:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Binary Classification @ -5.0 Threshold")
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            st.markdown(metric_card("Precision", f"{thresh.get('precision', 0):.4f}", "cyan"), unsafe_allow_html=True)
        with bc2:
            st.markdown(metric_card("Recall", f"{thresh.get('recall', 0):.4f}", "amber"), unsafe_allow_html=True)
        with bc3:
            st.markdown(metric_card("F1 Score", f"{thresh.get('f1', 0):.4f}", "green"), unsafe_allow_html=True)
        with bc4:
            st.markdown(metric_card("AUC", f"{thresh.get('auc', 0):.4f}", "purple"), unsafe_allow_html=True)


# ===================================================================
# Page 6 — Feature Importance
# ===================================================================

elif page == "Feature Importance":
    st.markdown("## Feature Importance")
    st.markdown('<p style="color:#8B949E; margin-top:-10px;">SHAP-based explainability analysis</p>',
                unsafe_allow_html=True)

    results = load_pipeline_results()
    models_available = available_models()
    feature_names = load_feature_names()
    scaler = load_scaler()
    df = load_dataset()

    # Show pre-computed global importance from pipeline results
    if results and "explainability" in results:
        top_features = results["explainability"].get("top_features", [])
        if top_features:
            st.markdown("#### Global Feature Importance (Top 10)")

            tf_df = pd.DataFrame(top_features)
            fig = go.Figure(go.Bar(
                x=tf_df["mean_abs_shap"],
                y=tf_df["feature"],
                orientation="h",
                marker=dict(
                    color=tf_df["mean_abs_shap"],
                    colorscale=[[0, "#00D4FF"], [0.5, "#7C5CFC"], [1, "#FF3366"]],
                ),
                text=[f"{v:.3f}" for v in tf_df["mean_abs_shap"]],
                textposition="auto",
                textfont=dict(color="white"),
            ))
            fig.update_layout(
                xaxis_title="Mean |SHAP Value|",
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                font=dict(color="#E6EDF3", family="Inter"),
                xaxis=dict(gridcolor="#1a2332"),
                yaxis_gridcolor="#1a2332",
                margin=dict(l=250, r=20, t=10, b=50),
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Interactive SHAP computation
    if models_available and feature_names and df is not None:
        st.markdown("#### Interactive SHAP Analysis")

        ctl1, ctl2 = st.columns([1, 1])
        with ctl1:
            model_name = st.selectbox("Model", models_available, key="fi_model")
        with ctl2:
            sample_size = st.slider("Sample size", 50, 500, 200, step=50)

        model = load_model(model_name)
        if model is None:
            st.error(f"Cannot load model: {model_name}")
            st.stop()

        if st.button("Compute SHAP Values", type="primary"):
            with st.spinner("Computing SHAP values (this may take a moment)..."):
                try:
                    from orbital_sentinel.preprocessing.cleaning import clean_dataframe
                    from orbital_sentinel.features import engineer_all_features
                    from orbital_sentinel.explainability.shap_analysis import (
                        compute_shap_values,
                        global_feature_importance,
                        top_features_for_prediction,
                    )

                    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
                    sample_clean = clean_dataframe(sample_df)
                    sample_feat = engineer_all_features(sample_clean)
                    for col in feature_names:
                        if col not in sample_feat.columns:
                            sample_feat[col] = 0.0
                    X_sample = sample_feat[feature_names].values

                    if scaler is not None:
                        X_sample = scaler.transform(X_sample)

                    shap_result = compute_shap_values(model, X_sample, feature_names, max_samples=sample_size)

                    # Global importance
                    global_imp = global_feature_importance(shap_result["shap_values"], feature_names, top_k=20)
                    imp_df = pd.DataFrame(global_imp)

                    fig_imp = go.Figure(go.Bar(
                        x=imp_df["mean_abs_shap"],
                        y=imp_df["feature"],
                        orientation="h",
                        marker=dict(
                            color=imp_df["mean_abs_shap"],
                            colorscale=[[0, "#00D4FF"], [0.5, "#7C5CFC"], [1, "#FF3366"]],
                        ),
                    ))
                    fig_imp.update_layout(
                        xaxis_title="Mean |SHAP Value|",
                        yaxis=dict(autorange="reversed"),
                        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                        font=dict(color="#E6EDF3", family="Inter"),
                        xaxis=dict(gridcolor="#1a2332"),
                        yaxis_gridcolor="#1a2332",
                        margin=dict(l=250, r=20, t=10, b=50),
                        height=600,
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

                    # Individual prediction
                    st.markdown("##### Individual Prediction Explanation")
                    idx = st.number_input("Sample index", 0, max(0, len(shap_result["shap_values"]) - 1), 0)

                    top_local = top_features_for_prediction(
                        shap_result["shap_values"], feature_names, idx=int(idx), top_k=15,
                    )
                    tl_df = pd.DataFrame(top_local)

                    bar_colors = ["#FF3366" if v > 0 else "#00D4FF" for v in tl_df["shap_value"]]
                    fig_local = go.Figure(go.Bar(
                        x=tl_df["shap_value"], y=tl_df["feature"],
                        orientation="h", marker_color=bar_colors,
                    ))
                    fig_local.update_layout(
                        xaxis_title="SHAP Value",
                        yaxis=dict(autorange="reversed"),
                        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                        font=dict(color="#E6EDF3", family="Inter"),
                        xaxis=dict(gridcolor="#1a2332"),
                        yaxis_gridcolor="#1a2332",
                        margin=dict(l=250, r=20, t=10, b=50),
                        height=500,
                    )
                    st.plotly_chart(fig_local, use_container_width=True)
                    st.caption("Red = pushes risk higher | Blue = pushes risk lower")

                except Exception as e:
                    st.error(f"SHAP computation failed: {e}")
    else:
        st.info("Model, feature names, and dataset required for interactive SHAP.")


# ===================================================================
# Page 7 — Physics Lab
# ===================================================================

elif page == "Physics Lab":
    st.markdown("## Physics Verification Lab")
    st.markdown('<p style="color:#8B949E; margin-top:-10px;">First-principles collision probability analysis</p>',
                unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.warning("Dataset not found.")
        st.stop()

    event_ids = sorted(df["event_id"].unique())[:500] if "event_id" in df.columns else list(range(min(500, len(df))))
    selected_event = st.selectbox("Select Conjunction Event", event_ids, key="phys_event")

    event_rows = df[df["event_id"] == selected_event] if "event_id" in df.columns else df.iloc[[selected_event]]
    if event_rows.empty:
        st.error("No data for selected event.")
        st.stop()

    row = event_rows.iloc[-1]
    row_dict = row.to_dict()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    with st.spinner("Running physics analysis..."):
        try:
            from orbital_sentinel.physics import verify_conjunction_physics
            physics_result = verify_conjunction_physics(row_dict)
        except Exception as e:
            st.error(f"Physics verification failed: {e}")
            st.stop()

    # Geometry
    st.markdown("#### Conjunction Geometry")
    geo = physics_result.get("geometry", {})

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(metric_card("Miss Distance", f"{geo.get('miss_distance_km', 0):.4f} km", "cyan"),
                    unsafe_allow_html=True)
    with g2:
        st.markdown(metric_card("Relative Speed", f"{geo.get('relative_speed_km_s', 0):.3f} km/s", "purple"),
                    unsafe_allow_html=True)
    with g3:
        st.markdown(metric_card("Approach Angle", f"{geo.get('approach_angle_deg', 0):.1f} deg", "amber"),
                    unsafe_allow_html=True)
    with g4:
        st.markdown(metric_card("Encounter Type", geo.get("encounter_type", "N/A")), unsafe_allow_html=True)

    g5, g6, g7, g8 = st.columns(4)
    with g5:
        st.markdown(metric_card("Miss R", f"{geo.get('miss_r_km', 0):.4f} km"), unsafe_allow_html=True)
    with g6:
        st.markdown(metric_card("Miss T", f"{geo.get('miss_t_km', 0):.4f} km"), unsafe_allow_html=True)
    with g7:
        st.markdown(metric_card("Miss N", f"{geo.get('miss_n_km', 0):.4f} km"), unsafe_allow_html=True)
    with g8:
        st.markdown(metric_card("Duration", f"{geo.get('encounter_duration_s', 0):.1f} s"), unsafe_allow_html=True)

    # Collision Probability
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### Collision Probability")

    cp1, cp2, cp3 = st.columns(3)
    pc = physics_result.get("analytic_pc", 0)
    log_pc = physics_result.get("log10_pc", float("nan"))
    with cp1:
        st.markdown(metric_card("Analytic Pc", f"{pc:.2e}" if np.isfinite(pc) else "N/A", "cyan"),
                    unsafe_allow_html=True)
    with cp2:
        st.markdown(metric_card("log10(Pc)", f"{log_pc:.2f}" if np.isfinite(log_pc) else "N/A",
                                "red" if np.isfinite(log_pc) and log_pc > -5 else "green"),
                    unsafe_allow_html=True)
    with cp3:
        se = physics_result.get("short_encounter_valid", False)
        st.markdown(metric_card("Short Encounter", "VALID" if se else "INVALID",
                                "green" if se else "red"), unsafe_allow_html=True)

    # Covariance validation
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### Covariance Validation")
    cov_valid = physics_result.get("covariance_valid", False)
    if cov_valid:
        st.markdown(alert_card("Covariance matrices pass all validation checks", "All eigenvalues positive, matrices symmetric", "info"),
                    unsafe_allow_html=True)
    else:
        for w in physics_result.get("covariance_warnings", ["Validation issues detected"]):
            st.markdown(alert_card("Covariance Warning", str(w), "warning"), unsafe_allow_html=True)

    # Physical consistency
    checks = physics_result.get("physical_consistency", [])
    if checks:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Physical Consistency Checks")
        for c in checks:
            passed = c.get("passed", False)
            check_name = c.get("check", "")
            detail = c.get("detail", "")
            lvl = "info" if passed else "critical"
            status = "PASS" if passed else "FAIL"
            st.markdown(alert_card(f"[{status}] {check_name}", detail, lvl), unsafe_allow_html=True)

    # Monte Carlo
    mc = physics_result.get("mc_verification")
    if mc and "error" not in mc:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Monte Carlo Verification")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(metric_card("MC Pc", f"{mc.get('mc_pc', 0):.2e}", "cyan"), unsafe_allow_html=True)
        with mc2:
            st.markdown(metric_card("Analytic Pc", f"{mc.get('analytic_pc', 0):.2e}", "purple"),
                        unsafe_allow_html=True)
        with mc3:
            agree = mc.get("agreement", False)
            st.markdown(metric_card("Agreement", "YES" if agree else "NO", "green" if agree else "red"),
                        unsafe_allow_html=True)



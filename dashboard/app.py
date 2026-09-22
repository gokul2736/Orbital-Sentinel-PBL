"""Orbital Sentinel — Space Situational Awareness Dashboard."""

import sys
import json
import math
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
    page_icon="\U0001F6F0️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-primary: #06090f;
    --bg-secondary: #0c1117;
    --bg-card: #111820;
    --bg-card-hover: #161f2d;
    --bg-glass: rgba(16,22,32,0.72);
    --bg-glass-heavy: rgba(12,17,23,0.88);
    --border: #1a2332;
    --border-light: #243044;
    --border-glow: rgba(0,212,255,0.08);
    --text-primary: #E8EDF4;
    --text-secondary: #6e7d8f;
    --text-muted: #3d4f63;
    --accent-cyan: #00D4FF;
    --accent-green: #00E87B;
    --accent-amber: #FFAA00;
    --accent-red: #FF2D55;
    --accent-purple: #7B61FF;
    --accent-blue: #3B82F6;
    --glow-cyan: rgba(0,212,255,0.10);
    --glow-red: rgba(255,45,85,0.10);
    --glow-green: rgba(0,232,123,0.10);
    --glow-purple: rgba(123,97,255,0.10);
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --shadow-sm: 0 2px 8px rgba(0,0,0,0.2);
    --shadow-md: 0 4px 24px rgba(0,0,0,0.3);
    --shadow-lg: 0 8px 40px rgba(0,0,0,0.4);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, header[data-testid="stHeader"], footer,
div[data-testid="stToolbar"], div[data-testid="stDecoration"],
.stDeployButton, #stDecoration { display: none !important; }

/* ── Base layout ── */
.main .block-container {
    padding-top: 0.5rem;
    padding-bottom: 2rem;
    max-width: 1480px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
    color: var(--text-primary) !important;
}
h2 { font-size: 1.45rem !important; }
h4 { font-size: 1.0rem !important; }

p, span, div, label { font-family: 'Inter', sans-serif; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-light); }

/* ── Metric cards (glassmorphism v2) ── */
.metric-card {
    background: var(--bg-glass);
    backdrop-filter: blur(16px) saturate(140%);
    -webkit-backdrop-filter: blur(16px) saturate(140%);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    margin-bottom: 10px;
    transition: all 0.3s cubic-bezier(0.22,1,0.36,1);
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 5%, rgba(0,212,255,0.18) 50%, transparent 95%);
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 2px;
    background: transparent;
    transition: background 0.3s ease;
    border-radius: 2px;
}
.metric-card:hover {
    border-color: var(--border-light);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md), 0 0 0 1px var(--border-light);
}
.metric-card:hover::after { background: var(--accent-cyan); }
.metric-card .metric-label {
    font-size: 0.6rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 7px;
}
.metric-card .metric-value {
    font-size: 1.55rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
    line-height: 1.15;
}
.metric-card .metric-value.cyan { color: var(--accent-cyan); text-shadow: 0 0 24px var(--glow-cyan); }
.metric-card .metric-value.green { color: var(--accent-green); text-shadow: 0 0 24px var(--glow-green); }
.metric-card .metric-value.amber { color: var(--accent-amber); text-shadow: 0 0 24px rgba(255,170,0,0.15); }
.metric-card .metric-value.red { color: var(--accent-red); text-shadow: 0 0 24px var(--glow-red); }
.metric-card .metric-value.purple { color: var(--accent-purple); text-shadow: 0 0 24px var(--glow-purple); }

/* ── Risk badges ── */
.risk-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 5px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}
.risk-badge.high { background: rgba(255,45,85,0.12); color: #FF2D55; border: 1px solid rgba(255,45,85,0.25); }
.risk-badge.medium { background: rgba(255,170,0,0.12); color: #FFAA00; border: 1px solid rgba(255,170,0,0.25); }
.risk-badge.low { background: rgba(0,212,255,0.10); color: #00D4FF; border: 1px solid rgba(0,212,255,0.20); }
.risk-badge.negligible { background: rgba(0,232,123,0.10); color: #00E87B; border: 1px solid rgba(0,232,123,0.20); }

/* ── Alert cards ── */
.alert-card {
    background: var(--bg-glass);
    backdrop-filter: blur(10px);
    border-left: 3px solid;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 13px 18px;
    margin-bottom: 8px;
    font-family: 'Inter', sans-serif;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.alert-card:hover { transform: translateX(2px); box-shadow: var(--shadow-sm); }
.alert-card.critical { border-left-color: #FF2D55; background: linear-gradient(90deg, rgba(255,45,85,0.05), var(--bg-glass)); }
.alert-card.warning { border-left-color: #FFAA00; background: linear-gradient(90deg, rgba(255,170,0,0.05), var(--bg-glass)); }
.alert-card.info { border-left-color: #00D4FF; background: linear-gradient(90deg, rgba(0,212,255,0.04), var(--bg-glass)); }

/* ── Status dot ── */
.status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-dot.active {
    background: #00E87B;
    box-shadow: 0 0 6px rgba(0,232,123,0.6), 0 0 16px rgba(0,232,123,0.2);
    animation: pulse-glow 2s ease-in-out infinite;
}
.status-dot.warning {
    background: #FFAA00;
    box-shadow: 0 0 6px rgba(255,170,0,0.6);
    animation: pulse-glow 1.5s ease-in-out infinite;
}
.status-dot.critical {
    background: #FF2D55;
    box-shadow: 0 0 6px rgba(255,45,85,0.6);
    animation: pulse-glow 1s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
}

/* ── Section divider ── */
.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border) 30%, var(--border) 70%, transparent);
    margin: 22px 0;
}

/* ── Sidebar ── */
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c12 0%, #0c1117 40%, #0e1420 100%) !important;
    border-right: 1px solid var(--border) !important;
}
div[data-testid="stSidebar"] > div:first-child { padding-top: 0.4rem; }
div[data-testid="stSidebar"] .stRadio > label { display: none !important; }
div[data-testid="stSidebar"] .stRadio > div { gap: 1px !important; }
div[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 9px 14px !important;
    margin: 0 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    font-family: 'Inter', sans-serif !important;
    border-left: 2px solid transparent !important;
}
div[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(0,212,255,0.04) !important;
    color: var(--text-primary) !important;
    border-left-color: rgba(0,212,255,0.3) !important;
}
div[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
div[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: rgba(0,212,255,0.07) !important;
    color: var(--accent-cyan) !important;
    border-left: 2px solid var(--accent-cyan) !important;
    font-weight: 600 !important;
}
div[data-testid="stSidebar"] .stRadio > div > label > div:first-child {
    display: none !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 3px;
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 3px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm);
    padding: 7px 16px;
    font-weight: 500;
    font-size: 0.8rem;
    color: var(--text-secondary);
    background: transparent;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text-primary); }
.stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.08) !important;
    color: var(--accent-cyan) !important;
    font-weight: 600 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(123,97,255,0.08)) !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
    color: var(--accent-cyan) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.25s cubic-bezier(0.22,1,0.36,1) !important;
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 24px var(--glow-cyan), 0 2px 12px rgba(0,0,0,0.3) !important;
    transform: translateY(-1px) !important;
    background: linear-gradient(135deg, rgba(0,212,255,0.18), rgba(123,97,255,0.12)) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.18), rgba(0,212,255,0.08)) !important;
}

/* ── Inputs / Selects ── */
.stSelectbox > div > div,
.stSlider > div,
.stMultiSelect > div > div,
.stNumberInput > div > div {
    font-family: 'Inter', sans-serif !important;
}
div[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── DataFrame styling ── */
.stDataFrame {
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--border) !important;
}

/* ── Plotly chart containers ── */
.stPlotlyChart {
    border-radius: var(--radius-md);
    overflow: hidden;
}

/* ── Header bar ── */
.os-header {
    background: linear-gradient(90deg, var(--bg-secondary), rgba(0,212,255,0.02) 50%, var(--bg-secondary));
    border-bottom: 1px solid var(--border);
    padding: 8px 0;
    margin: -6px -1rem 14px -1rem;
    position: relative;
}
.os-header::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 10%;
    right: 10%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.12), transparent);
}
.os-header-inner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 8px;
}
.os-header-left {
    display: flex;
    align-items: center;
    gap: 14px;
}
.os-header-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.os-header-right {
    display: flex;
    align-items: center;
    gap: 22px;
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-muted);
}
.os-header-status {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--accent-green);
    font-weight: 500;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
}

/* ── Page subtitle ── */
.page-subtitle {
    color: var(--text-secondary);
    font-size: 0.86rem;
    margin-bottom: 18px;
    font-weight: 400;
    letter-spacing: 0.01em;
}

/* ── Section header with accent ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}
.section-header .sh-icon {
    width: 28px;
    height: 28px;
    border-radius: 7px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
}
.section-header .sh-text {
    font-size: 1.0rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}

/* ── Stat pill ── */
.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
}
.stat-pill.green { background: rgba(0,232,123,0.10); color: #00E87B; }
.stat-pill.red { background: rgba(255,45,85,0.10); color: #FF2D55; }
.stat-pill.cyan { background: rgba(0,212,255,0.10); color: #00D4FF; }
.stat-pill.amber { background: rgba(255,170,0,0.10); color: #FFAA00; }

/* ── Live indicator ── */
.live-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    background: rgba(255,45,85,0.10);
    border: 1px solid rgba(255,45,85,0.25);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    color: #FF2D55;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.live-tag .live-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #FF2D55;
    animation: pulse-glow 1s ease-in-out infinite;
}

/* ── Panel (glass container) ── */
.glass-panel {
    background: var(--bg-glass);
    backdrop-filter: blur(16px) saturate(140%);
    -webkit-backdrop-filter: blur(16px) saturate(140%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.glass-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 10%, rgba(0,212,255,0.15) 50%, transparent 90%);
}

/* ── Orbit ring animation (decorative) ── */
@keyframes orbit-spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
.orbit-ring {
    position: absolute;
    border: 1px solid rgba(0,212,255,0.06);
    border-radius: 50%;
    pointer-events: none;
}

/* ── Gradient text ── */
.gradient-text {
    background: linear-gradient(135deg, #00D4FF 0%, #7B61FF 50%, #FF2D55 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Data table (professional) ── */
.stDataFrame [data-testid="stDataFrameResizable"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}

/* ── Smooth fade-in for content ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.stMarkdown, .stPlotlyChart, .stDataFrame {
    animation: fadeSlideUp 0.35s ease-out;
}

/* ── Number highlight ── */
.num-highlight {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 2.2rem;
    line-height: 1;
}

/* ── Mini stat row ── */
.mini-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(26,35,50,0.5);
    font-size: 0.82rem;
}
.mini-stat:last-child { border-bottom: none; }
.mini-stat .ms-label { color: var(--text-secondary); font-weight: 500; }
.mini-stat .ms-value { color: var(--text-primary); font-family: 'JetBrains Mono', monospace; font-weight: 500; }

/* ── Conjunction card (enhanced) ── */
.conj-card {
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    margin-bottom: 10px;
    transition: all 0.3s cubic-bezier(0.22,1,0.36,1);
    position: relative;
}
.conj-card:hover {
    border-color: var(--border-light);
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}
.conj-card .conj-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.conj-card .conj-objects {
    font-weight: 700;
    color: var(--text-primary);
    font-size: 0.92rem;
}
.conj-card .conj-meta {
    display: flex;
    gap: 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-secondary);
}
.conj-card .conj-meta span { display: flex; align-items: center; gap: 4px; }
.conj-card .conj-meta .val { color: var(--text-primary); font-weight: 500; }

/* ── Progress bar mini ── */
.mini-progress {
    height: 3px;
    border-radius: 2px;
    background: var(--border);
    overflow: hidden;
    margin-top: 6px;
}
.mini-progress .bar {
    height: 100%;
    border-radius: 2px;
    transition: width 0.5s ease;
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
        <span style="color: #6e7d8f; font-size: 0.85rem;">{detail}</span>
    </div>
    """


def risk_color(val: float) -> str:
    if val > -5: return "#FF2D55"
    if val > -7: return "#FFAA00"
    if val > -15: return "#00D4FF"
    return "#00E87B"


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
    <div style="text-align:center; padding: 14px 0 4px;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.5rem; font-weight:600;
                    color:#3d4f63; letter-spacing:0.18em; text-transform:uppercase; margin-bottom:5px;">
            SPACE SITUATIONAL AWARENESS
        </div>
        <div style="font-size:1.15rem; font-weight:800; letter-spacing:0.06em;
                    font-family:'Inter',sans-serif;
                    background: linear-gradient(135deg, #00D4FF, #7B61FF);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;">
            ORBITAL SENTINEL
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.52rem; color:#3d4f63;
                    letter-spacing:0.08em; margin-top:4px;">
            CONJUNCTION RISK INTELLIGENCE v2.0
        </div>
        <div style="margin-top:8px; height:1px;
                    background:linear-gradient(90deg, transparent, #1a2332, transparent);"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin:10px 14px 4px; font-family:'JetBrains Mono',monospace; font-size:0.48rem;
                font-weight:600; color:#3d4f63; letter-spacing:0.18em; text-transform:uppercase;">
        OPERATIONS
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAV",
        [
            "Mission Control",
            "Live CDM Feed",
            "Risk Assessment",
            "Event Timeline",
            "Orbit Simulation",
            "Data Explorer",
            "Model Performance",
            "Feature Importance",
            "Physics Lab",
        ],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div style="margin: 14px 14px 0; padding-top:12px; border-top:1px solid #1a2332;">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
            <span class="status-dot active"></span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.62rem; font-weight:500; color:#00E87B;">
                ALL SYSTEMS NOMINAL
            </span>
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.5rem; color:#3d4f63; line-height:2.0;">
            <div style="display:flex; justify-content:space-between;">
                <span>ENGINE</span><span style="color:#6e7d8f;">XGBoost 2.0</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span>DATA</span><span style="color:#6e7d8f;">Space-Track + ESA</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span>MODELS</span><span style="color:#6e7d8f;">4 Ensemble</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span>PHYSICS</span><span style="color:#6e7d8f;">Monte Carlo</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header bar (renders on every page)
# ---------------------------------------------------------------------------
import datetime as _dt

_now = _dt.datetime.now(_dt.timezone.utc)
_page_upper = page.upper().replace(" ", " / ", 1) if page != "Mission Control" else "MISSION CONTROL"
st.markdown(f"""
<div class="os-header">
    <div class="os-header-inner">
        <div class="os-header-left">
            <span class="os-header-title">ORBITAL SENTINEL</span>
            <span style="color:#1a2332; font-size:0.6rem;">|</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.65rem;
                         font-weight:500; color:#00D4FF; letter-spacing:0.06em;">{_page_upper}</span>
        </div>
        <div class="os-header-right">
            <span style="color:#3d4f63;">{_now.strftime('%Y-%m-%d')}</span>
            <span style="color:#6e7d8f; font-weight:500;">{_now.strftime('%H:%M:%S')} UTC</span>
            <span class="os-header-status">
                <span class="status-dot active"></span> ONLINE
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ===================================================================
# Page 1 — Mission Control (Overview Dashboard)
# ===================================================================

if page == "Mission Control":
    st.markdown('<p class="page-subtitle">Operational overview of active conjunction events and system status</p>',
                unsafe_allow_html=True)

    df = load_dataset()
    results = load_pipeline_results()

    # Live data quick-fetch for Mission Control
    _mc_live_cdms = st.session_state.get("mc_live_cdms")
    _mc_high_count = 0
    _mc_total = 0
    if _mc_live_cdms:
        import math as _math
        _mc_total = len(_mc_live_cdms)
        for _c in _mc_live_cdms:
            _pc = float(_c.get("PC", 0) or 0)
            if _pc > 0 and _math.log10(_pc) > -5:
                _mc_high_count += 1

    # Hero metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _total_cdms = f"{len(df):,}" if df is not None else "N/A"
        st.markdown(metric_card("Training CDMs", _total_cdms, "cyan"), unsafe_allow_html=True)
    with c2:
        _n_ev = df["event_id"].nunique() if df is not None and "event_id" in df.columns else 0
        st.markdown(metric_card("Unique Events", f"{_n_ev:,}", "purple"), unsafe_allow_html=True)
    with c3:
        if _mc_live_cdms:
            st.markdown(metric_card("Live High-Risk", f"{_mc_high_count}", "red"), unsafe_allow_html=True)
        elif df is not None and "risk" in df.columns:
            _hr = int((df["risk"] > -5).sum())
            st.markdown(metric_card("High Risk", f"{_hr:,}", "red"), unsafe_allow_html=True)
        else:
            st.markdown(metric_card("High Risk", "N/A"), unsafe_allow_html=True)
    with c4:
        if results:
            _rmse = results.get("evaluation", {}).get("overall", {}).get("rmse", "N/A")
            st.markdown(metric_card("Best RMSE", f"{_rmse:.3f}" if isinstance(_rmse, float) else "N/A", "green"),
                        unsafe_allow_html=True)
        else:
            st.markdown(metric_card("Best RMSE", "N/A"), unsafe_allow_html=True)
    with c5:
        if results:
            _r2 = results.get("evaluation", {}).get("overall", {}).get("r2", "N/A")
            st.markdown(metric_card("Model R-Squared", f"{_r2:.4f}" if isinstance(_r2, float) else "N/A", "green"),
                        unsafe_allow_html=True)
        else:
            st.markdown(metric_card("Model R-Squared", "N/A"), unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Live CDM quick panel
    _mc_col_left, _mc_col_right = st.columns([1, 3])
    with _mc_col_left:
        if st.button("Refresh Live Data", type="primary", use_container_width=True):
            try:
                from orbital_sentinel.ingestion import SpaceTrackClient
                with SpaceTrackClient() as _client:
                    _live = _client.fetch_cdm(limit=10, enrich_gp=True)
                st.session_state["mc_live_cdms"] = _live
                st.rerun()
            except Exception as _e:
                st.error(f"Fetch failed: {_e}")
    with _mc_col_right:
        if _mc_live_cdms:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;">
                <span class="live-tag"><span class="live-dot"></span> LIVE FEED</span>
                <span class="stat-pill cyan">{_mc_total} CDMs loaded</span>
                <span class="stat-pill {'red' if _mc_high_count > 0 else 'green'}">{_mc_high_count} high-risk</span>
                <span style="color:#3d4f63; font-size:0.72rem; font-family:'JetBrains Mono',monospace;">
                    Source: Space-Track.org / 18th SDS
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display:flex; align-items:center; gap:14px;">
                <span style="color:#3d4f63; font-size:0.78rem;">
                    Click <strong style="color:#00D4FF;">Refresh Live Data</strong> to pull real-time conjunctions from Space-Track
                </span>
            </div>
            """, unsafe_allow_html=True)

    # Show live CDM summary if available
    if _mc_live_cdms:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Active Conjunction Events")
        import math as _math
        for _idx, _cdm in enumerate(_mc_live_cdms[:8]):
            _pc = float(_cdm.get("PC", 0) or 0)
            _risk_v = _math.log10(_pc) if _pc > 0 else -30
            _miss = float(_cdm.get("MIN_RNG", 0) or 0)
            _s1 = _cdm.get("SAT_1_NAME", "Unknown")
            _s2 = _cdm.get("SAT_2_NAME", "Unknown")
            _tca = str(_cdm.get("TCA", ""))[:16]
            _emg = _cdm.get("EMERGENCY_REPORTABLE", "N")
            _lvl = "critical" if _risk_v > -5 else ("warning" if _risk_v > -7 else "info")
            _badge = risk_badge(risk_label(_risk_v))
            st.markdown(f"""
            <div class="alert-card {_lvl}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <strong style="color:#E8EDF4;">{_s1}</strong>
                        <span style="color:#3d4f63; font-size:0.75rem; margin:0 6px;">vs</span>
                        <strong style="color:#E8EDF4;">{_s2}</strong>
                    </div>
                    {_badge}
                </div>
                <div style="margin-top:5px; display:flex; gap:20px; font-family:'JetBrains Mono',monospace;
                            font-size:0.72rem; color:#6e7d8f;">
                    <span>TCA: <span style="color:#E8EDF4;">{_tca}</span></span>
                    <span>Miss: <span style="color:{'#FF2D55' if _miss < 100 else '#FFAA00' if _miss < 500 else '#E8EDF4'}">{_miss:,.1f} km</span></span>
                    <span>Pc: <span style="color:{'#FF2D55' if _risk_v > -5 else '#E8EDF4'}">{_pc:.2e}</span></span>
                    {'<span style="color:#FF2D55;">EMERGENCY</span>' if _emg == "Y" else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if df is None:
        st.info("Training dataset not found. The live feed above works independently. Run the pipeline to enable the charts below.")
        st.stop()

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

            fig.add_vline(x=-5, line_dash="dash", line_color="#FF2D55", line_width=2,
                          annotation_text="HIGH", annotation_font_color="#FF2D55")
            fig.add_vline(x=-7, line_dash="dash", line_color="#FFAA00", line_width=2,
                          annotation_text="MEDIUM", annotation_font_color="#FFAA00")

            fig.update_layout(
                xaxis_title="log10(Collision Probability)",
                yaxis_title="Count",
                paper_bgcolor="#06090f", plot_bgcolor="#06090f",
                font=dict(color="#E8EDF4", family="Inter"),
                xaxis=dict(gridcolor="#111820"),
                yaxis=dict(gridcolor="#111820"),
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
        cat_colors = ["#00E87B", "#00D4FF", "#FFAA00", "#FF2D55"]

        cc1, cc2 = st.columns([1, 2])
        with cc1:
            fig_pie = go.Figure(go.Pie(
                labels=list(cat_data.keys()),
                values=list(cat_data.values()),
                marker=dict(colors=cat_colors, line=dict(color="#06090f", width=2)),
                textinfo="percent+label",
                textfont=dict(size=11),
                hole=0.55,
            ))
            fig_pie.update_layout(
                paper_bgcolor="#06090f",
                font=dict(color="#E8EDF4", family="Inter"),
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
                        <span style="color:#E8EDF4; font-weight:500;">{cat_name}</span><br>
                        <span style="color:#6e7d8f; font-size:0.85rem;">{count:,} events ({pct:.1f}%)</span>
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
    st.markdown('<p class="page-subtitle">Full conjunction analysis with ML + Physics fusion</p>',
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
            colors_bar = ["#FF2D55" if v > 0 else "#00D4FF" for v in tf_df["shap_value"]]
            fig_shap = go.Figure(go.Bar(
                x=tf_df["shap_value"], y=tf_df["feature"],
                orientation="h", marker_color=colors_bar,
            ))
            fig_shap.update_layout(
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="#06090f", plot_bgcolor="#06090f",
                font=dict(color="#E8EDF4", family="Inter"),
                xaxis=dict(title="SHAP Value", gridcolor="#111820"),
                yaxis_gridcolor="#111820",
                margin=dict(l=200, r=20, t=10, b=40),
                height=300,
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("Red = increases risk | Blue = decreases risk")


# ===================================================================
# Page 4 — Event Timeline
# ===================================================================

elif page == "Event Timeline":
    st.markdown('<p style="color:#7a8899; font-size:0.85rem; margin-bottom:16px;">Risk evolution as TCA approaches</p>',
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
                line=dict(color="#FF2D55", width=2), marker=dict(size=7),
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
        st.markdown('<p style="color:#6e7d8f;">Events where risk is increasing and exceeds threshold</p>',
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
            fig_matrix.add_hline(y=-5.0, line_dash="dash", line_color="#FF2D55",
                                 annotation_text="High-risk threshold")
            fig_matrix.update_layout(height=400, margin=dict(t=20, b=30), template="plotly_dark")
            st.plotly_chart(fig_matrix, use_container_width=True)


# ===================================================================
# Page 5 — Model Performance
# ===================================================================

elif page == "Model Performance":
    st.markdown('<p style="color:#7a8899; font-size:0.85rem; margin-bottom:16px;">Training results and model comparison</p>',
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
            border_color = "#00E87B" if is_best else "#30363D"
            badge = '<span style="color:#00E87B; font-size:0.7rem; font-weight:600;">BEST</span>' if is_best else ""
            st.markdown(f"""
            <div style="background:#1C2333; border:2px solid {border_color}; border-radius:12px; padding:20px; text-align:center;">
                <div style="font-size:1.1rem; font-weight:600; color:#E8EDF4;">{name} {badge}</div>
                <div style="margin-top:12px;">
                    <div style="color:#6e7d8f; font-size:0.7rem; text-transform:uppercase;">RMSE</div>
                    <div style="font-family:'JetBrains Mono'; font-size:1.5rem; color:{'#00E87B' if is_best else '#00D4FF'};">
                        {val.get('rmse', 0):.3f}
                    </div>
                </div>
                <div style="margin-top:8px;">
                    <div style="color:#6e7d8f; font-size:0.7rem; text-transform:uppercase;">R-SQUARED</div>
                    <div style="font-family:'JetBrains Mono'; font-size:1.2rem; color:#E8EDF4;">
                        {val.get('r2', 0):.4f}
                    </div>
                </div>
                <div style="margin-top:8px;">
                    <div style="color:#6e7d8f; font-size:0.7rem; text-transform:uppercase;">MAE</div>
                    <div style="font-family:'JetBrains Mono'; font-size:1.2rem; color:#E8EDF4;">
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
            colors = ["#00E87B" if n == best_name else "#00D4FF" for n in names]

            fig = go.Figure(go.Bar(
                x=names, y=values,
                marker_color=colors,
                text=[f"{v:.4f}" for v in values],
                textposition="auto",
                textfont=dict(color="white", size=14),
            ))
            fig.update_layout(
                yaxis_title=metric_key.upper(),
                paper_bgcolor="#06090f", plot_bgcolor="#06090f",
                font=dict(color="#E8EDF4", family="Inter"),
                xaxis=dict(gridcolor="#111820"),
                yaxis=dict(gridcolor="#111820"),
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
    st.markdown('<p style="color:#7a8899; font-size:0.85rem; margin-bottom:16px;">SHAP-based explainability analysis</p>',
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
                    colorscale=[[0, "#00D4FF"], [0.5, "#7C5CFC"], [1, "#FF2D55"]],
                ),
                text=[f"{v:.3f}" for v in tf_df["mean_abs_shap"]],
                textposition="auto",
                textfont=dict(color="white"),
            ))
            fig.update_layout(
                xaxis_title="Mean |SHAP Value|",
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="#06090f", plot_bgcolor="#06090f",
                font=dict(color="#E8EDF4", family="Inter"),
                xaxis=dict(gridcolor="#111820"),
                yaxis_gridcolor="#111820",
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
                            colorscale=[[0, "#00D4FF"], [0.5, "#7C5CFC"], [1, "#FF2D55"]],
                        ),
                    ))
                    fig_imp.update_layout(
                        xaxis_title="Mean |SHAP Value|",
                        yaxis=dict(autorange="reversed"),
                        paper_bgcolor="#06090f", plot_bgcolor="#06090f",
                        font=dict(color="#E8EDF4", family="Inter"),
                        xaxis=dict(gridcolor="#111820"),
                        yaxis_gridcolor="#111820",
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

                    bar_colors = ["#FF2D55" if v > 0 else "#00D4FF" for v in tl_df["shap_value"]]
                    fig_local = go.Figure(go.Bar(
                        x=tl_df["shap_value"], y=tl_df["feature"],
                        orientation="h", marker_color=bar_colors,
                    ))
                    fig_local.update_layout(
                        xaxis_title="SHAP Value",
                        yaxis=dict(autorange="reversed"),
                        paper_bgcolor="#06090f", plot_bgcolor="#06090f",
                        font=dict(color="#E8EDF4", family="Inter"),
                        xaxis=dict(gridcolor="#111820"),
                        yaxis_gridcolor="#111820",
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
    st.markdown('<p style="color:#7a8899; font-size:0.85rem; margin-bottom:16px;">First-principles collision probability analysis</p>',
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


# ===================================================================
# Page — 3D Orbit Simulation
# ===================================================================

elif page == "Orbit Simulation":
    st.markdown('<p style="color:#7a8899; font-size:0.85rem; margin-bottom:16px;">3D orbital visualization of conjunction events around Earth</p>',
                unsafe_allow_html=True)

    from orbital_sentinel.visualization.orbit_3d import (
        create_orbit_figure,
        create_conjunction_detail_figure,
    )

    df = load_dataset()
    if df is None:
        st.warning("Dataset not found. Run the ingestion pipeline first.")
        st.stop()

    # ── View selector ──
    view_mode = st.radio(
        "View", ["Orbital Environment", "Conjunction Detail"], horizontal=True, key="sim_view",
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if view_mode == "Orbital Environment":
        # Controls
        ctl1, ctl2, ctl3 = st.columns([1, 1, 1])
        with ctl1:
            n_events = st.slider("Events to display", 2, 12, 5, key="sim_n")
        with ctl2:
            risk_filter = st.selectbox("Filter by risk", ["All", "High only", "Medium+", "Low+"], key="sim_filter")
        with ctl3:
            st.markdown(metric_card("Dataset Events",
                                    f"{df['event_id'].nunique():,}" if "event_id" in df.columns else "N/A",
                                    "cyan"), unsafe_allow_html=True)

        # Filter events
        last_cdm = df.groupby("event_id").last().reset_index() if "event_id" in df.columns else df
        if risk_filter == "High only" and "risk" in last_cdm.columns:
            last_cdm = last_cdm[last_cdm["risk"] > -5]
        elif risk_filter == "Medium+" and "risk" in last_cdm.columns:
            last_cdm = last_cdm[last_cdm["risk"] > -7]
        elif risk_filter == "Low+" and "risk" in last_cdm.columns:
            last_cdm = last_cdm[last_cdm["risk"] > -15]

        if len(last_cdm) == 0:
            st.info("No events match this filter.")
            st.stop()

        sample_df = last_cdm.sample(n=min(n_events, len(last_cdm)), random_state=42)

        # Render 3D globe
        with st.spinner("Rendering orbital environment..."):
            fig_env = create_orbit_figure(events_df=sample_df, show_all_orbits=True, dark_theme=True)
        st.plotly_chart(fig_env, use_container_width=True, key="orbit_env")

        # Legend
        st.markdown("""
        <div style="display:flex; gap:24px; justify-content:center; padding:8px 0;">
            <span style="color:#00D4FF; font-size:0.8rem;">&#9473; Target Orbits</span>
            <span style="color:#FF2D55; font-size:0.8rem;">- - Chaser Orbits</span>
            <span style="color:#FFAA00; font-size:0.8rem;">&#9670; Conjunction Points</span>
            <span style="color:#00E87B; font-size:0.8rem;">&#9679; Earth (6,378 km)</span>
        </div>
        """, unsafe_allow_html=True)

        # Event summary table
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Displayed Conjunctions")

        for _, row in sample_df.iterrows():
            eid = row.get("event_id", "?")
            risk_val = row.get("risk", -30)
            miss = row.get("miss_distance", 0)
            speed = row.get("relative_speed", 0)
            t_alt = row.get("t_h_per", 0)
            c_alt = row.get("c_h_per", 0)
            lvl = "critical" if risk_val > -5 else ("warning" if risk_val > -7 else "info")

            st.markdown(f"""
            <div class="alert-card {lvl}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong>Event {eid}</strong>
                    {risk_badge(risk_label(risk_val))}
                </div>
                <span style="color:#6e7d8f; font-size:0.85rem;">
                    Risk: {risk_val:.1f} | Miss: {miss:,.0f} m | Speed: {speed:,.0f} m/s |
                    Target alt: {t_alt:,.0f} km | Chaser alt: {c_alt:,.0f} km
                </span>
            </div>
            """, unsafe_allow_html=True)

    else:
        # ── Single conjunction detail ──
        event_ids = sorted(df["event_id"].unique())[:500] if "event_id" in df.columns else []
        if not event_ids:
            st.info("No events available.")
            st.stop()

        ctl1, ctl2 = st.columns([2, 1])
        with ctl1:
            selected = st.selectbox("Select Conjunction Event", event_ids, key="sim_detail")
        with ctl2:
            event_rows = df[df["event_id"] == selected] if "event_id" in df.columns else df.head(1)
            st.markdown(metric_card("CDMs for Event", str(len(event_rows)), "cyan"), unsafe_allow_html=True)

        row = event_rows.iloc[-1]
        row_dict = row.to_dict()

        # Render detailed conjunction
        with st.spinner("Rendering conjunction geometry..."):
            fig_detail = create_conjunction_detail_figure(row_dict, dark_theme=True)
        st.plotly_chart(fig_detail, use_container_width=True, key="orbit_detail")

        st.markdown("""
        <div style="display:flex; gap:24px; justify-content:center; padding:8px 0;">
            <span style="color:#00D4FF; font-size:0.8rem;">&#9473; Target Orbit</span>
            <span style="color:#FF2D55; font-size:0.8rem;">- - Chaser Orbit</span>
            <span style="color:#FFAA00; font-size:0.8rem;">&#8230; Miss Vector</span>
            <span style="color:rgba(0,212,255,0.5); font-size:0.8rem;">&#9711; Covariance</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Orbital parameters — target vs chaser side by side
        st.markdown("#### Orbital Elements")
        left, right = st.columns(2)

        with left:
            st.markdown("""
            <div style="text-align:center; padding:4px 0 8px;">
                <span style="color:#00D4FF; font-weight:600; font-size:0.9rem; letter-spacing:0.05em;">
                    TARGET OBJECT
                </span>
            </div>
            """, unsafe_allow_html=True)
            t1, t2 = st.columns(2)
            with t1:
                st.markdown(metric_card("Semi-Major Axis", f"{row.get('t_j2k_sma', 0):,.1f} km", "cyan"), unsafe_allow_html=True)
                st.markdown(metric_card("Inclination", f"{row.get('t_j2k_inc', 0):.2f} deg", "cyan"), unsafe_allow_html=True)
            with t2:
                st.markdown(metric_card("Eccentricity", f"{row.get('t_j2k_ecc', 0):.6f}"), unsafe_allow_html=True)
                st.markdown(metric_card("Perigee Alt", f"{row.get('t_h_per', 0):,.0f} km"), unsafe_allow_html=True)

        with right:
            st.markdown("""
            <div style="text-align:center; padding:4px 0 8px;">
                <span style="color:#FF2D55; font-weight:600; font-size:0.9rem; letter-spacing:0.05em;">
                    CHASER OBJECT
                </span>
            </div>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(metric_card("Semi-Major Axis", f"{row.get('c_j2k_sma', 0):,.1f} km", "red"), unsafe_allow_html=True)
                st.markdown(metric_card("Inclination", f"{row.get('c_j2k_inc', 0):.2f} deg", "red"), unsafe_allow_html=True)
            with c2:
                st.markdown(metric_card("Eccentricity", f"{row.get('c_j2k_ecc', 0):.6f}"), unsafe_allow_html=True)
                st.markdown(metric_card("Perigee Alt", f"{row.get('c_h_per', 0):,.0f} km"), unsafe_allow_html=True)

        # Encounter metrics
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Encounter Metrics")

        e1, e2, e3, e4 = st.columns(4)
        risk_val = row.get("risk", -30)
        with e1:
            st.markdown(metric_card("Miss Distance", f"{row.get('miss_distance', 0):,.0f} m", "amber"), unsafe_allow_html=True)
        with e2:
            st.markdown(metric_card("Relative Speed", f"{row.get('relative_speed', 0):,.0f} m/s", "purple"), unsafe_allow_html=True)
        with e3:
            st.markdown(metric_card("Collision Risk", f"{risk_val:.2f}",
                                    "red" if risk_val > -5 else ("amber" if risk_val > -7 else "green")),
                        unsafe_allow_html=True)
        with e4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Level</div>
                <div style="padding-top:4px;">{risk_badge(risk_label(risk_val))}</div>
            </div>
            """, unsafe_allow_html=True)

        # Relative position breakdown
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Relative Position at TCA (RTN Frame)")
        rtn_data = {
            "Component": ["Radial (R)", "Along-Track (T)", "Cross-Track (N)"],
            "Position (m)": [
                row.get("relative_position_r", 0),
                row.get("relative_position_t", 0),
                row.get("relative_position_n", 0),
            ],
            "Velocity (m/s)": [
                row.get("relative_velocity_r", 0),
                row.get("relative_velocity_t", 0),
                row.get("relative_velocity_n", 0),
            ],
        }
        rtn_df = pd.DataFrame(rtn_data)
        fig_rtn = go.Figure()
        fig_rtn.add_trace(go.Bar(
            x=rtn_df["Component"], y=rtn_df["Position (m)"],
            name="Position (m)", marker_color="#00D4FF", opacity=0.85,
            text=[f"{v:,.0f}" for v in rtn_df["Position (m)"]],
            textposition="auto", textfont=dict(color="white"),
        ))
        fig_rtn.update_layout(
            height=280, paper_bgcolor="#06090f", plot_bgcolor="#06090f",
            font=dict(color="#E8EDF4", family="Inter"),
            yaxis=dict(title="Distance (m)", gridcolor="#111820"),
            xaxis=dict(gridcolor="#111820"),
            margin=dict(l=50, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_rtn, use_container_width=True)


# ===================================================================
# Page — Live CDM Feed
# ===================================================================

elif page == "Live CDM Feed":
    st.markdown('<p style="color:#7a8899; font-size:0.85rem; margin-bottom:16px;">Fetch real conjunction data from Space-Track.org and run predictions</p>',
                unsafe_allow_html=True)

    models_available = available_models()
    feature_names = load_feature_names()
    scaler = load_scaler()

    has_model = models_available and feature_names is not None

    # Status bar
    s1, s2, s3 = st.columns(3)
    with s1:
        api_configured = True
        try:
            from dotenv import load_dotenv as _ld
            import os as _os
            _ld()
            api_configured = bool(_os.getenv("SPACE_TRACK_USERNAME")) and bool(_os.getenv("SPACE_TRACK_PASSWORD"))
        except Exception:
            api_configured = False
        status = "CONFIGURED" if api_configured else "NOT CONFIGURED"
        st.markdown(metric_card("API Status", status, "green" if api_configured else "red"), unsafe_allow_html=True)
    with s2:
        st.markdown(metric_card("ML Model", models_available[0] if models_available else "NOT LOADED",
                                "green" if models_available else "red"), unsafe_allow_html=True)
    with s3:
        st.markdown(metric_card("Features", str(len(feature_names)) if feature_names else "N/A",
                                "cyan"), unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Fetch controls
    ctl1, ctl2, ctl3 = st.columns([1, 1, 1])
    with ctl1:
        n_cdms = st.selectbox("CDMs to fetch", [1, 5, 10, 20], index=1)
    with ctl2:
        st.markdown("")
        fetch_clicked = st.button("Fetch Live CDMs", type="primary", disabled=not api_configured, use_container_width=True)
    with ctl3:
        cdm_count = st.session_state.get("live_cdms_count", 0)
        st.markdown(metric_card("CDMs Loaded", str(cdm_count), "green" if cdm_count > 0 else ""), unsafe_allow_html=True)

    if fetch_clicked:
        with st.spinner("Connecting to Space-Track.org..."):
            try:
                from orbital_sentinel.ingestion.cdm_api import SpaceTrackClient
                from orbital_sentinel.ingestion.cdm_mapper import map_cdm_to_esa_format

                client = SpaceTrackClient()
                client.login()
                raw_cdms = client.fetch_cdm(limit=n_cdms)
                client.close()

                st.session_state["live_cdms_raw"] = raw_cdms
                st.session_state["live_cdms_count"] = len(raw_cdms)

            except Exception as e:
                st.error(f"Space-Track fetch failed: {e}")
                st.session_state["live_cdms_raw"] = None

    raw_cdms = st.session_state.get("live_cdms_raw")

    if raw_cdms:
        st.markdown(alert_card(
            f"Fetched {len(raw_cdms)} live CDM(s) from Space-Track",
            "Real conjunction data from the 18th Space Defense Squadron",
            "info",
        ), unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Show raw CDM fields
        st.markdown("#### Raw CDM Data")
        st.markdown('<p style="color:#6e7d8f; margin-top:-10px;">Original fields from the Space-Track CDM Public endpoint + GP orbital enrichment</p>',
                    unsafe_allow_html=True)
        with st.expander("View raw Space-Track CDM fields", expanded=False):
            for i, cdm in enumerate(raw_cdms[:5]):
                st.markdown(f"**CDM #{i+1} — {cdm.get('SAT_1_NAME', 'N/A')} vs {cdm.get('SAT_2_NAME', 'N/A')}**")
                key_fields = {
                    "CDM_ID": cdm.get("CDM_ID"),
                    "TCA": cdm.get("TCA"),
                    "MIN_RNG (km)": cdm.get("MIN_RNG"),
                    "PC (collision prob)": cdm.get("PC"),
                    "Target": cdm.get("SAT_1_NAME"),
                    "Chaser": cdm.get("SAT_2_NAME"),
                    "Target Type": cdm.get("SAT1_OBJECT_TYPE"),
                    "Chaser Type": cdm.get("SAT2_OBJECT_TYPE"),
                    "Target RCS": cdm.get("SAT1_RCS"),
                    "Chaser RCS": cdm.get("SAT2_RCS"),
                    "Emergency": cdm.get("EMERGENCY_REPORTABLE"),
                    "Created": cdm.get("CREATED"),
                    "GP Enriched": cdm.get("_enriched_gp", False),
                    "Target SMA (km)": cdm.get("OBJECT1_SEMI_MAJOR_AXIS"),
                    "Target Inclination": cdm.get("OBJECT1_INCLINATION"),
                    "Chaser SMA (km)": cdm.get("OBJECT2_SEMI_MAJOR_AXIS"),
                    "Chaser Inclination": cdm.get("OBJECT2_INCLINATION"),
                }
                st.json({k: v for k, v in key_fields.items() if v is not None})

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Map, predict, and simulate
        st.markdown("#### Live CDM Analysis & 3D Simulation")
        st.markdown('<p style="color:#6e7d8f; margin-top:-10px;">Each CDM mapped, scored, and visualized as an interactive 3D conjunction simulation</p>',
                    unsafe_allow_html=True)
        from orbital_sentinel.ingestion.cdm_mapper import map_cdm_to_esa_format
        from orbital_sentinel.visualization.orbit_3d import (
            create_cdm_3d_simulation,
            create_bplane_view,
        )

        # Summary table of all fetched CDMs
        st.markdown("#### Live Conjunction Summary")
        import math as _math
        _summary_rows = []
        for _cdm in raw_cdms:
            _m = map_cdm_to_esa_format(_cdm)
            _pc = float(_cdm.get("PC", 0) or 0)
            _risk_val = _math.log10(_pc) if _pc > 0 else -30.0
            _miss_km = float(_cdm.get("MIN_RNG", 0) or _m.get("miss_distance", 0))
            _cat = "HIGH" if _risk_val > -5 else ("MEDIUM" if _risk_val > -7 else ("LOW" if _risk_val > -15 else "NEGLIGIBLE"))
            _summary_rows.append({
                "Target": _cdm.get("SAT_1_NAME", "Unknown"),
                "Chaser": _cdm.get("SAT_2_NAME", "Unknown"),
                "TCA (UTC)": str(_cdm.get("TCA", ""))[:16],
                "Miss (km)": f"{_miss_km:,.1f}",
                "Pc": f"{_pc:.2e}" if _pc > 0 else "N/A",
                "log10(Pc)": f"{_risk_val:.2f}" if _risk_val > -29 else "N/A",
                "Risk": _cat,
                "Emergency": _cdm.get("EMERGENCY_REPORTABLE", "N"),
            })
        _summary_df = pd.DataFrame(_summary_rows)
        st.dataframe(_summary_df, use_container_width=True, hide_index=True, height=min(400, 38 + 35 * len(_summary_rows)))

        # High-risk alert count
        _high_count = sum(1 for r in _summary_rows if r["Risk"] == "HIGH")
        if _high_count > 0:
            st.markdown(alert_card(
                f"{_high_count} HIGH-RISK conjunction(s) detected",
                "These events have collision probability > 10^-5 — potential maneuver candidates",
                "critical",
            ), unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Individual CDM analysis
        st.markdown("#### Detailed CDM Analysis & 3D Simulation")

        for i, cdm in enumerate(raw_cdms):
            mapped = map_cdm_to_esa_format(cdm)
            sat1_name = cdm.get("SAT_1_NAME", cdm.get("OBJECT1_OBJECT_NAME", "Object 1"))
            sat2_name = cdm.get("SAT_2_NAME", cdm.get("OBJECT2_OBJECT_NAME", "Object 2"))
            _miss_km = float(cdm.get("MIN_RNG", 0) or mapped.get("miss_distance", 0))
            _pc_val = float(cdm.get("PC", 0) or 0)
            _risk_v = mapped.get("risk", -30)
            if isinstance(_risk_v, float) and np.isnan(_risk_v):
                _risk_v = -30
            _lvl = "critical" if _risk_v > -5 else ("warning" if _risk_v > -7 else "info")

            st.markdown(f"""
            <div class="alert-card {_lvl}" style="border-left-width: 4px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong>CDM #{i+1} — {sat1_name} vs {sat2_name}</strong>
                    {risk_badge(risk_label(_risk_v))}
                </div>
                <span style="color:#6e7d8f; font-size:0.85rem;">
                    TCA: {cdm.get('TCA', 'N/A')[:16]} |
                    Miss: {_miss_km:,.1f} km |
                    Pc: {_pc_val:.2e} |
                    GP Enriched: {'Yes' if cdm.get('_enriched_gp') else 'No'}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Tabs: Prediction | 3D Simulation | B-Plane | Orbital Data
            cdm_tab1, cdm_tab2, cdm_tab3, cdm_tab4 = st.tabs([
                "Risk Prediction", "3D Orbit Simulation", "B-Plane View", "Orbital Data",
            ])

            with cdm_tab1:
                if has_model:
                    try:
                        model = load_model(models_available[0])
                        from orbital_sentinel.inference.pipeline import predict_single_event
                        result = predict_single_event(mapped, model, scaler, feature_names,
                                                      include_physics=True, include_explanation=True)

                        p1, p2, p3, p4 = st.columns(4)
                        with p1:
                            st.markdown(metric_card("Predicted Risk", f"{result['prediction']:.2f}",
                                                    "red" if result["prediction"] > -5 else "cyan"), unsafe_allow_html=True)
                        with p2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Category</div>
                                <div style="padding-top:4px;">{risk_badge(result['risk_category'])}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with p3:
                            conf = result.get("confidence", 0)
                            st.markdown(metric_card("Confidence", f"{conf:.2f}" if conf else "N/A", "green"),
                                        unsafe_allow_html=True)
                        with p4:
                            interval = result.get("interval")
                            iv_text = f"[{interval[0]:.1f}, {interval[1]:.1f}]" if interval else "N/A"
                            st.markdown(metric_card("90% Interval", iv_text, "purple"), unsafe_allow_html=True)

                        if result.get("explanation"):
                            st.markdown(alert_card("AI Assessment", result["explanation"], "info"), unsafe_allow_html=True)

                        physics = result.get("physics_result")
                        if physics:
                            ph1, ph2, ph3 = st.columns(3)
                            with ph1:
                                log_pc = physics.get("log10_pc", float("nan"))
                                st.markdown(metric_card("Physics Pc",
                                                        f"10^{log_pc:.2f}" if np.isfinite(log_pc) else "N/A"),
                                            unsafe_allow_html=True)
                            with ph2:
                                st.markdown(metric_card("Covariance",
                                                        "VALID" if physics.get("covariance_valid") else "INVALID",
                                                        "green" if physics.get("covariance_valid") else "red"),
                                            unsafe_allow_html=True)
                            with ph3:
                                fusion = result.get("fusion", {})
                                fused = fusion.get("fused_risk", result["prediction"])
                                st.markdown(metric_card("Fused Risk", f"{fused:.2f}",
                                                        "red" if fused > -5 else "cyan"),
                                            unsafe_allow_html=True)

                    except Exception as e:
                        st.markdown(alert_card(f"CDM #{i+1} Prediction Error", str(e), "warning"), unsafe_allow_html=True)
                else:
                    st.info("Load a trained model to enable ML predictions on live data.")

            with cdm_tab2:
                st.markdown(f"""
                <div style="text-align:center; padding:4px 0 12px;">
                    <span style="color:#00D4FF; font-size:0.85rem; font-weight:500; letter-spacing:0.04em;">
                        ANIMATED CONJUNCTION APPROACH — {sat1_name} vs {sat2_name}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                with st.spinner("Rendering 3D conjunction simulation..."):
                    fig_sim = create_cdm_3d_simulation(
                        mapped,
                        sat1_name=sat1_name,
                        sat2_name=sat2_name,
                        n_frames=60,
                        dark_theme=True,
                    )
                st.plotly_chart(fig_sim, use_container_width=True)

                st.markdown("""
                <div style="background:rgba(28,35,51,0.6); border:1px solid #1a2332; border-radius:8px;
                            padding:10px 16px; margin-top:4px; font-size:0.8rem; color:#6e7d8f;">
                    <strong style="color:#00D4FF;">Controls:</strong>
                    Press <strong>Play Approach</strong> to animate satellites approaching TCA.
                    Use <strong>Jump to TCA</strong> to see the conjunction point with miss vector
                    and covariance ellipsoids. Drag to rotate, scroll to zoom.
                </div>
                """, unsafe_allow_html=True)

                sim_m1, sim_m2, sim_m3, sim_m4 = st.columns(4)
                with sim_m1:
                    st.markdown(metric_card("Target SMA", f"{mapped.get('t_j2k_sma', 0):,.1f} km", "cyan"),
                                unsafe_allow_html=True)
                with sim_m2:
                    st.markdown(metric_card("Chaser SMA", f"{mapped.get('c_j2k_sma', 0):,.1f} km", "red"),
                                unsafe_allow_html=True)
                with sim_m3:
                    _miss_val = mapped.get("miss_distance", 0)
                    _miss_display = f"{_miss_val:,.1f} km" if mapped.get("_source") == "cdm_public" else f"{_miss_val:,.0f} m"
                    st.markdown(metric_card("Miss Distance", _miss_display, "amber"),
                                unsafe_allow_html=True)
                with sim_m4:
                    risk_val = mapped.get("risk")
                    if risk_val is not None and not (isinstance(risk_val, float) and np.isnan(risk_val)):
                        st.markdown(metric_card("Risk", f"{float(risk_val):.2f}",
                                                "red" if float(risk_val) > -5 else "green"),
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(metric_card("Risk", "Pending ML", "amber"), unsafe_allow_html=True)

            with cdm_tab3:
                st.markdown(f"""
                <div style="text-align:center; padding:4px 0 12px;">
                    <span style="color:#FFAA00; font-size:0.85rem; font-weight:500; letter-spacing:0.04em;">
                        B-PLANE CLOSE APPROACH GEOMETRY
                    </span>
                </div>
                """, unsafe_allow_html=True)

                fig_bp = create_bplane_view(mapped, dark_theme=True)
                st.plotly_chart(fig_bp, use_container_width=True)

                bp1, bp2, bp3 = st.columns(3)
                with bp1:
                    st.markdown(metric_card("Radial Miss",
                                            f"{mapped.get('relative_position_r', 0):,.4f} km", "cyan"),
                                unsafe_allow_html=True)
                with bp2:
                    st.markdown(metric_card("Along-Track Miss",
                                            f"{mapped.get('relative_position_t', 0):,.4f} km", "purple"),
                                unsafe_allow_html=True)
                with bp3:
                    st.markdown(metric_card("Cross-Track Miss",
                                            f"{mapped.get('relative_position_n', 0):,.4f} km", "amber"),
                                unsafe_allow_html=True)

            with cdm_tab4:
                obj_left, obj_right = st.columns(2)
                with obj_left:
                    st.markdown(f"""
                    <div style="text-align:center; padding:4px 0 8px;">
                        <span style="color:#00D4FF; font-weight:600; font-size:0.85rem;">
                            TARGET — {sat1_name}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(metric_card("SMA", f"{mapped.get('t_j2k_sma', 0):,.1f} km", "cyan"), unsafe_allow_html=True)
                    st.markdown(metric_card("Eccentricity", f"{mapped.get('t_j2k_ecc', 0):.6f}"), unsafe_allow_html=True)
                    st.markdown(metric_card("Inclination", f"{mapped.get('t_j2k_inc', 0):.2f} deg"), unsafe_allow_html=True)
                    st.markdown(metric_card("RAAN", f"{mapped.get('t_j2k_raan', 0):.2f} deg"), unsafe_allow_html=True)
                    st.markdown(metric_card("Arg Perigee", f"{mapped.get('t_j2k_argp', 0):.2f} deg"), unsafe_allow_html=True)
                with obj_right:
                    st.markdown(f"""
                    <div style="text-align:center; padding:4px 0 8px;">
                        <span style="color:#FF2D55; font-weight:600; font-size:0.85rem;">
                            CHASER — {sat2_name}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(metric_card("SMA", f"{mapped.get('c_j2k_sma', 0):,.1f} km", "red"), unsafe_allow_html=True)
                    st.markdown(metric_card("Eccentricity", f"{mapped.get('c_j2k_ecc', 0):.6f}"), unsafe_allow_html=True)
                    st.markdown(metric_card("Inclination", f"{mapped.get('c_j2k_inc', 0):.2f} deg"), unsafe_allow_html=True)
                    st.markdown(metric_card("RAAN", f"{mapped.get('c_j2k_raan', 0):.2f} deg"), unsafe_allow_html=True)
                    st.markdown(metric_card("Arg Perigee", f"{mapped.get('c_j2k_argp', 0):.2f} deg"), unsafe_allow_html=True)

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("##### Relative State at TCA")
                rtn_data = {
                    "Component": ["Radial (R)", "Along-Track (T)", "Cross-Track (N)"],
                    "Position (km)": [
                        mapped.get("relative_position_r", 0),
                        mapped.get("relative_position_t", 0),
                        mapped.get("relative_position_n", 0),
                    ],
                    "Velocity (km/s)": [
                        mapped.get("relative_velocity_r", 0),
                        mapped.get("relative_velocity_t", 0),
                        mapped.get("relative_velocity_n", 0),
                    ],
                }
                st.dataframe(pd.DataFrame(rtn_data), use_container_width=True, hide_index=True)

            if i < len(raw_cdms) - 1:
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        if not has_model:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown(alert_card(
                "Model Not Loaded",
                "Run the training pipeline to enable ML predictions on live CDM data.",
                "warning",
            ), unsafe_allow_html=True)

    elif not fetch_clicked:
        st.markdown("""
        <div style="background: linear-gradient(135deg, var(--bg-card) 0%, rgba(0,212,255,0.03) 50%, var(--bg-secondary) 100%);
                    border: 1px solid var(--border); border-radius: 12px;
                    text-align:center; padding:48px 24px; margin-top:12px;">
            <div style="font-size:3rem; opacity:0.9;">&#127760;</div>
            <div style="margin-top:14px; font-size:1.15rem; font-weight:600; color:#E8EDF4;">
                Live Conjunction Data Feed
            </div>
            <div style="color:#6e7d8f; font-size:0.88rem; margin-top:8px; max-width:600px; margin-left:auto; margin-right:auto;">
                Click <strong style="color:#00D4FF;">Fetch Live CDMs</strong> to pull real-time
                Conjunction Data Messages from the 18th Space Defense Squadron via Space-Track.org.
                Each CDM is automatically enriched with GP orbital elements for full 3D simulation.
            </div>
            <div style="margin-top:16px; display:flex; gap:24px; justify-content:center; flex-wrap:wrap;">
                <div style="background:rgba(0,212,255,0.08); border:1px solid rgba(0,212,255,0.2);
                            border-radius:8px; padding:10px 18px;">
                    <div style="color:#00D4FF; font-size:0.7rem; font-weight:600; text-transform:uppercase;
                                letter-spacing:0.08em;">Source</div>
                    <div style="color:#E8EDF4; font-size:0.85rem; font-weight:500;">Space-Track.org</div>
                </div>
                <div style="background:rgba(0,255,136,0.08); border:1px solid rgba(0,255,136,0.2);
                            border-radius:8px; padding:10px 18px;">
                    <div style="color:#00E87B; font-size:0.7rem; font-weight:600; text-transform:uppercase;
                                letter-spacing:0.08em;">Data Class</div>
                    <div style="color:#E8EDF4; font-size:0.85rem; font-weight:500;">CDM Public + GP</div>
                </div>
                <div style="background:rgba(124,92,252,0.08); border:1px solid rgba(124,92,252,0.2);
                            border-radius:8px; padding:10px 18px;">
                    <div style="color:#7C5CFC; font-size:0.7rem; font-weight:600; text-transform:uppercase;
                                letter-spacing:0.08em;">Features</div>
                    <div style="color:#E8EDF4; font-size:0.85rem; font-weight:500;">ML + Physics + 3D Viz</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ===================================================================
# Page — CDM Data Explorer
# ===================================================================

elif page == "Data Explorer":
    st.markdown('<p style="color:#7a8899; font-size:0.85rem; margin-bottom:16px;">Browse the ESA Kelvins conjunction dataset — 162K real CDMs</p>',
                unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.warning("Dataset not found. Run the ingestion pipeline first.")
        st.stop()

    results = load_pipeline_results()

    # Dataset overview
    st.markdown("#### Dataset Overview")
    o1, o2, o3, o4, o5 = st.columns(5)
    with o1:
        st.markdown(metric_card("Total Rows", f"{len(df):,}", "cyan"), unsafe_allow_html=True)
    with o2:
        st.markdown(metric_card("Columns", str(df.shape[1]), "purple"), unsafe_allow_html=True)
    with o3:
        n_events = df["event_id"].nunique() if "event_id" in df.columns else 0
        st.markdown(metric_card("Events", f"{n_events:,}", "green"), unsafe_allow_html=True)
    with o4:
        mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.markdown(metric_card("Memory", f"{mem_mb:.1f} MB", "amber"), unsafe_allow_html=True)
    with o5:
        null_pct = df.isnull().mean().mean() * 100
        st.markdown(metric_card("Null %", f"{null_pct:.2f}%", "green" if null_pct < 5 else "red"),
                    unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    exp_tab1, exp_tab2, exp_tab3, exp_tab4 = st.tabs([
        "Sample Rows", "Column Stats", "Distributions", "Data Quality",
    ])

    with exp_tab1:
        st.markdown("#### Sample CDM Records")
        st.markdown('<p style="color:#6e7d8f;">Each row is a real Conjunction Data Message from ESA</p>',
                    unsafe_allow_html=True)

        sample_mode = st.radio("Sample", ["First 50", "Random 50", "High-risk only"], horizontal=True, key="exp_sample")
        if sample_mode == "First 50":
            sample = df.head(50)
        elif sample_mode == "Random 50":
            sample = df.sample(50, random_state=42)
        else:
            sample = df[df["risk"] > -5].head(50) if "risk" in df.columns else df.head(50)

        key_cols = ["event_id", "time_to_tca", "risk", "miss_distance", "relative_speed",
                    "t_j2k_sma", "t_j2k_inc", "c_j2k_sma", "c_j2k_inc", "c_object_type"]
        avail_cols = [c for c in key_cols if c in sample.columns]
        st.dataframe(sample[avail_cols], use_container_width=True, hide_index=True, height=400)

        with st.expander("View all columns for first row"):
            first_row = df.iloc[0]
            col_data = pd.DataFrame({
                "Column": first_row.index,
                "Value": first_row.values,
                "Type": [str(df[c].dtype) for c in first_row.index],
            })
            st.dataframe(col_data, use_container_width=True, hide_index=True, height=400)

    with exp_tab2:
        st.markdown("#### Column Statistics")
        st.markdown('<p style="color:#6e7d8f;">Descriptive statistics for selected numeric columns</p>',
                    unsafe_allow_html=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = st.multiselect(
            "Select columns", numeric_cols,
            default=["risk", "miss_distance", "relative_speed", "time_to_tca", "t_j2k_sma"][:min(5, len(numeric_cols))],
            key="exp_cols",
        )

        if selected_cols:
            stats = df[selected_cols].describe().T
            stats["null_count"] = df[selected_cols].isnull().sum()
            stats["null_pct"] = (df[selected_cols].isnull().mean() * 100).round(2)
            st.dataframe(
                stats.style.format("{:.4f}", subset=["mean", "std", "min", "25%", "50%", "75%", "max"])
                     .format("{:,.0f}", subset=["count", "null_count"])
                     .format("{:.2f}%", subset=["null_pct"]),
                use_container_width=True,
            )

    with exp_tab3:
        st.markdown("#### Feature Distributions")
        st.markdown('<p style="color:#6e7d8f;">Histogram and box plot for any numeric feature</p>',
                    unsafe_allow_html=True)

        dist_col = st.selectbox(
            "Select column",
            ["risk", "miss_distance", "relative_speed", "time_to_tca",
             "t_j2k_sma", "t_j2k_ecc", "t_j2k_inc", "mahalanobis_distance",
             "c_j2k_sma", "c_j2k_ecc", "c_j2k_inc"],
            key="exp_dist",
        )

        if dist_col in df.columns:
            col_data = df[dist_col].dropna()

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.markdown(metric_card("Mean", f"{col_data.mean():.4f}"), unsafe_allow_html=True)
            with d2:
                st.markdown(metric_card("Median", f"{col_data.median():.4f}"), unsafe_allow_html=True)
            with d3:
                st.markdown(metric_card("Std Dev", f"{col_data.std():.4f}"), unsafe_allow_html=True)
            with d4:
                st.markdown(metric_card("Range", f"{col_data.min():.2f} to {col_data.max():.2f}"), unsafe_allow_html=True)

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=col_data, nbinsx=80,
                marker_color="#00D4FF", opacity=0.8,
                hovertemplate=f"{dist_col}: " + "%{x:.2f}<br>Count: %{y:,}<extra></extra>",
            ))
            fig_dist.update_layout(
                xaxis_title=dist_col, yaxis_title="Count",
                paper_bgcolor="#06090f", plot_bgcolor="#06090f",
                font=dict(color="#E8EDF4", family="Inter"),
                xaxis=dict(gridcolor="#111820"), yaxis=dict(gridcolor="#111820"),
                margin=dict(l=50, r=20, t=10, b=50), height=350,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # Box plot
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                x=col_data, name=dist_col,
                marker_color="#7C5CFC", line_color="#7C5CFC",
                boxmean="sd",
            ))
            fig_box.update_layout(
                paper_bgcolor="#06090f", plot_bgcolor="#06090f",
                font=dict(color="#E8EDF4", family="Inter"),
                xaxis=dict(gridcolor="#111820"), yaxis=dict(gridcolor="#111820"),
                margin=dict(l=50, r=20, t=10, b=40), height=200,
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning(f"Column '{dist_col}' not found in dataset.")

    with exp_tab4:
        st.markdown("#### Data Quality Assessment")
        st.markdown('<p style="color:#6e7d8f;">Validation results, outlier detection, and event structure</p>',
                    unsafe_allow_html=True)

        if results and "validation" in results:
            quality = results["validation"].get("quality", {})

            q1, q2, q3, q4 = st.columns(4)
            with q1:
                st.markdown(metric_card("Total Rows", f"{quality.get('total_rows', len(df)):,}", "cyan"),
                            unsafe_allow_html=True)
            with q2:
                st.markdown(metric_card("Risk Floor %", f"{quality.get('risk_floor_pct', 0):.1f}%", "amber"),
                            unsafe_allow_html=True)
            with q3:
                neg_miss = quality.get("negative_miss_distance_count", 0)
                st.markdown(metric_card("Neg Miss Dist", str(neg_miss), "green" if neg_miss == 0 else "red"),
                            unsafe_allow_html=True)
            with q4:
                neg_speed = quality.get("negative_speed_count", 0)
                st.markdown(metric_card("Neg Speed", str(neg_speed), "green" if neg_speed == 0 else "red"),
                            unsafe_allow_html=True)

            outlier_cols = quality.get("outlier_columns", [])
            if outlier_cols:
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("#### Columns with Outliers")
                st.markdown('<p style="color:#6e7d8f;">Columns where values exceed 3 standard deviations</p>',
                            unsafe_allow_html=True)

                n_per_row = 5
                for row_start in range(0, len(outlier_cols), n_per_row):
                    row_cols = outlier_cols[row_start:row_start + n_per_row]
                    cols = st.columns(n_per_row)
                    for j, col_name in enumerate(row_cols):
                        with cols[j]:
                            st.markdown(f"""
                            <div style="background:#1C2333; border:1px solid #FFAA00; border-radius:8px;
                                        padding:8px 12px; text-align:center; margin-bottom:8px;">
                                <span style="color:#FFAA00; font-size:0.8rem; font-family:'JetBrains Mono';">
                                    {col_name}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)

        # CDMs per event distribution
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### CDMs per Event")
        st.markdown('<p style="color:#6e7d8f;">Distribution of Conjunction Data Messages across events</p>',
                    unsafe_allow_html=True)
        if "event_id" in df.columns:
            cdm_counts = df["event_id"].value_counts()
            fig_cdm = go.Figure()
            fig_cdm.add_trace(go.Histogram(
                x=cdm_counts.values, nbinsx=50,
                marker_color="#00E87B", opacity=0.8,
            ))
            fig_cdm.update_layout(
                xaxis_title="Number of CDMs per Event", yaxis_title="Number of Events",
                paper_bgcolor="#06090f", plot_bgcolor="#06090f",
                font=dict(color="#E8EDF4", family="Inter"),
                xaxis=dict(gridcolor="#111820"), yaxis=dict(gridcolor="#111820"),
                margin=dict(l=50, r=20, t=10, b=50), height=300,
            )
            st.plotly_chart(fig_cdm, use_container_width=True)

            cdm_q1, cdm_q2, cdm_q3, cdm_q4 = st.columns(4)
            with cdm_q1:
                st.markdown(metric_card("Mean CDMs/Event", f"{cdm_counts.mean():.1f}", "cyan"), unsafe_allow_html=True)
            with cdm_q2:
                st.markdown(metric_card("Median", f"{cdm_counts.median():.0f}", "green"), unsafe_allow_html=True)
            with cdm_q3:
                st.markdown(metric_card("Max CDMs", str(cdm_counts.max()), "amber"), unsafe_allow_html=True)
            with cdm_q4:
                single = (cdm_counts == 1).sum()
                st.markdown(metric_card("Single-CDM Events", f"{single:,}", "purple"), unsafe_allow_html=True)

        # Feature correlation matrix
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Feature Correlation (Key Features)")
        st.markdown('<p style="color:#6e7d8f;">Pearson correlation between key numeric features</p>',
                    unsafe_allow_html=True)
        corr_cols = ["risk", "miss_distance", "relative_speed", "time_to_tca",
                     "t_j2k_sma", "t_j2k_ecc", "t_j2k_inc", "mahalanobis_distance"]
        avail_corr = [c for c in corr_cols if c in df.columns]
        if len(avail_corr) > 2:
            corr_matrix = df[avail_corr].corr()
            fig_corr = go.Figure(go.Heatmap(
                z=corr_matrix.values, x=avail_corr, y=avail_corr,
                colorscale=[
                    [0.0, "#FF2D55"], [0.25, "#7a2040"],
                    [0.5, "#1C2333"],
                    [0.75, "#1a4070"], [1.0, "#00D4FF"],
                ],
                zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont=dict(size=11, color="#E8EDF4"),
                colorbar=dict(
                    title="r", tickfont=dict(color="#6e7d8f"),
                    titlefont=dict(color="#6e7d8f"),
                ),
            ))
            fig_corr.update_layout(
                paper_bgcolor="#06090f", plot_bgcolor="#06090f",
                font=dict(color="#E8EDF4", family="Inter"),
                margin=dict(l=140, r=30, t=10, b=140), height=480,
                xaxis=dict(tickangle=-40, tickfont=dict(size=11)),
                yaxis=dict(tickfont=dict(size=11)),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

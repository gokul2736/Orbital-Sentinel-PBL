"""System Status dashboard page — health, module inventory, infrastructure."""

import datetime
import importlib
import json
import platform
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models_saved"
PROOFS_DIR = PROJECT_ROOT / "proofs"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "esa_kelvins"
SRC_DIR = PROJECT_ROOT / "src" / "orbital_sentinel"
TESTS_DIR = PROJECT_ROOT / "tests"


def _metric_card(label: str, value: str, color: str = "") -> str:
    color_class = f" {color}" if color else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value{color_class}">{value}</div>
    </div>
    """


def _alert_card(title: str, detail: str, level: str = "info") -> str:
    return f"""
    <div class="alert-card {level}">
        <strong>{title}</strong><br>
        <span style="color: #6e7d8f; font-size: 0.85rem;">{detail}</span>
    </div>
    """


def _check_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _get_version(package: str) -> str | None:
    try:
        mod = importlib.import_module(package)
        return getattr(mod, "__version__", "?")
    except Exception:
        return None


def _dir_size(path: Path) -> int:
    total = 0
    if path.exists():
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _count_files(path: Path, pattern: str = "*") -> int:
    if not path.exists():
        return 0
    return sum(1 for f in path.rglob(pattern) if f.is_file())


def _divider():
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ── Plotly layout defaults ──────────────────────────────────────────
_PLOTLY_BASE = dict(
    paper_bgcolor="#06090f",
    plot_bgcolor="#06090f",
    font=dict(color="#E8EDF4", family="Inter"),
    margin=dict(l=20, r=20, t=30, b=20),
)


def render_system_status():
    """Render the full System Status page."""

    st.markdown(
        '<p style="color:#7a8899; font-size:0.85rem; margin-bottom:16px;">'
        "System health, module inventory, and infrastructure status</p>",
        unsafe_allow_html=True,
    )

    # ──────────────────────────────────────────────────────────────────
    # 1. System Health Overview
    # ──────────────────────────────────────────────────────────────────
    st.markdown("#### System Health")

    api_ok = _check_import("fastapi")
    db_ok = _check_import("sqlalchemy")
    ml_ready = MODELS_DIR.exists() and any(MODELS_DIR.glob("*.joblib"))
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    h1, h2, h3, h4, h5 = st.columns(5)
    with h1:
        st.markdown(
            _metric_card("API Status", "ONLINE" if api_ok else "OFFLINE", "green" if api_ok else "red"),
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            _metric_card("Database", "ONLINE" if db_ok else "OFFLINE", "green" if db_ok else "red"),
            unsafe_allow_html=True,
        )
    with h3:
        st.markdown(
            _metric_card("ML Engine", "READY" if ml_ready else "NO MODELS", "green" if ml_ready else "amber"),
            unsafe_allow_html=True,
        )
    with h4:
        st.markdown(_metric_card("Pipeline", "IDLE", "cyan"), unsafe_allow_html=True)
    with h5:
        st.markdown(
            _metric_card("Clock (UTC)", now_utc.strftime("%H:%M:%S"), "purple"),
            unsafe_allow_html=True,
        )

    _divider()

    # ──────────────────────────────────────────────────────────────────
    # 2. Infrastructure Status
    # ──────────────────────────────────────────────────────────────────
    st.markdown("#### Infrastructure")

    inf1, inf2, inf3 = st.columns(3)

    # ── Runtime ──
    with inf1:
        st.markdown(
            f"""
            <div style="background:rgba(0,212,255,0.05); border:1px solid rgba(0,212,255,0.15);
                        border-radius:10px; padding:16px 20px;">
                <div style="color:#00D4FF; font-size:0.65rem; font-weight:700; text-transform:uppercase;
                            letter-spacing:0.1em; margin-bottom:12px;">Runtime</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#6e7d8f; line-height:2.2;">
                    <div style="display:flex; justify-content:space-between;">
                        <span>Python</span>
                        <span style="color:#E8EDF4;">{platform.python_version()}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span>OS</span>
                        <span style="color:#E8EDF4;">{platform.system()} {platform.release()}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span>Platform</span>
                        <span style="color:#E8EDF4;">{platform.machine()}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span>CWD</span>
                        <span style="color:#E8EDF4; font-size:0.65rem; max-width:160px; overflow:hidden;
                                     text-overflow:ellipsis; white-space:nowrap;"
                              title="{PROJECT_ROOT}">{PROJECT_ROOT.name}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Dependencies ──
    PACKAGES = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("streamlit", "streamlit"),
        ("plotly", "plotly"),
        ("fastapi", "fastapi"),
        ("sqlalchemy", "sqlalchemy"),
    ]

    with inf2:
        rows_html = ""
        for display_name, import_name in PACKAGES:
            ver = _get_version(import_name)
            dot_color = "#00E87B" if ver else "#FF2D55"
            ver_text = ver or "missing"
            rows_html += f"""
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="display:flex; align-items:center; gap:6px;">
                        <span style="display:inline-block; width:6px; height:6px; border-radius:50%;
                                     background:{dot_color}; box-shadow:0 0 5px {dot_color};"></span>
                        {display_name}
                    </span>
                    <span style="color:#E8EDF4;">{ver_text}</span>
                </div>"""

        st.markdown(
            f"""
            <div style="background:rgba(123,97,255,0.05); border:1px solid rgba(123,97,255,0.15);
                        border-radius:10px; padding:16px 20px;">
                <div style="color:#7B61FF; font-size:0.65rem; font-weight:700; text-transform:uppercase;
                            letter-spacing:0.1em; margin-bottom:12px;">Dependencies</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#6e7d8f;
                            line-height:2.1;">{rows_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Storage ──
    with inf3:
        models_count = _count_files(MODELS_DIR, "*.joblib")
        models_size = _dir_size(MODELS_DIR)
        data_count = _count_files(DATA_DIR)
        data_size = _dir_size(DATA_DIR)
        proofs_count = _count_files(PROOFS_DIR)
        proofs_size = _dir_size(PROOFS_DIR)

        st.markdown(
            f"""
            <div style="background:rgba(0,232,123,0.05); border:1px solid rgba(0,232,123,0.15);
                        border-radius:10px; padding:16px 20px;">
                <div style="color:#00E87B; font-size:0.65rem; font-weight:700; text-transform:uppercase;
                            letter-spacing:0.1em; margin-bottom:12px;">Storage</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#6e7d8f; line-height:2.2;">
                    <div style="display:flex; justify-content:space-between;">
                        <span>Models</span>
                        <span style="color:#E8EDF4;">{models_count} files &middot; {_fmt_bytes(models_size)}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span>Data</span>
                        <span style="color:#E8EDF4;">{data_count} files &middot; {_fmt_bytes(data_size)}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span>Proofs</span>
                        <span style="color:#E8EDF4;">{proofs_count} files &middot; {_fmt_bytes(proofs_size)}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-top:6px; padding-top:6px;
                                border-top:1px solid rgba(0,232,123,0.15);">
                        <span style="color:#00E87B;">Total</span>
                        <span style="color:#E8EDF4; font-weight:600;">
                            {_fmt_bytes(models_size + data_size + proofs_size)}
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _divider()

    # ──────────────────────────────────────────────────────────────────
    # 3. Module Inventory
    # ──────────────────────────────────────────────────────────────────
    st.markdown("#### Module Inventory")
    st.markdown(
        '<p style="color:#6e7d8f; font-size:0.8rem;">Import status of every orbital_sentinel sub-package</p>',
        unsafe_allow_html=True,
    )

    MODULES = [
        ("models.baselines", "ML Baselines"),
        ("models.temporal", "Temporal Models"),
        ("models.ensemble", "Ensemble"),
        ("features", "Feature Eng."),
        ("preprocessing", "Preprocessing"),
        ("physics", "Physics Engine"),
        ("ingestion", "CDM Ingestion"),
        ("evaluation", "Evaluation"),
        ("explainability", "Explainability"),
        ("uncertainty", "Uncertainty"),
        ("validation", "Validation"),
        ("api", "REST API"),
        ("cli", "CLI"),
        ("database", "Database"),
        ("alerts", "Alerts"),
        ("reporting", "Reporting"),
        ("visualization", "Visualization"),
        ("fusion", "Fusion"),
        ("monitoring", "Monitoring"),
        ("events", "Events"),
    ]

    cols_per_row = 4
    for row_start in range(0, len(MODULES), cols_per_row):
        row_modules = MODULES[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for idx, (mod_path, display) in enumerate(row_modules):
            full = f"orbital_sentinel.{mod_path}"
            ok = _check_import(full)
            dot = "#00E87B" if ok else "#FF2D55"
            label = "LOADED" if ok else "ERROR"
            label_color = "#00E87B" if ok else "#FF2D55"
            bg = "rgba(0,232,123,0.04)" if ok else "rgba(255,45,85,0.04)"
            border = "rgba(0,232,123,0.15)" if ok else "rgba(255,45,85,0.15)"

            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="background:{bg}; border:1px solid {border};
                                border-radius:8px; padding:10px 14px; margin-bottom:8px; min-height:72px;">
                        <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
                            <span style="display:inline-block; width:7px; height:7px; border-radius:50%;
                                         background:{dot}; box-shadow:0 0 6px {dot};"></span>
                            <span style="color:#E8EDF4; font-size:0.8rem; font-weight:600;">{display}</span>
                        </div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:{label_color};
                                    letter-spacing:0.08em;">{label}</div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:0.55rem; color:#3d4f63;
                                    margin-top:2px;">{full}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    _divider()

    # ──────────────────────────────────────────────────────────────────
    # 4. Model Registry
    # ──────────────────────────────────────────────────────────────────
    st.markdown("#### Model Registry")
    st.markdown(
        '<p style="color:#6e7d8f; font-size:0.8rem;">Trained model artifacts and performance metrics</p>',
        unsafe_allow_html=True,
    )

    metrics_path = MODELS_DIR / "model_metrics.json"
    model_metrics = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as mf:
                model_metrics = json.load(mf)
        except Exception:
            model_metrics = {}

    if MODELS_DIR.exists():
        joblib_files = sorted(MODELS_DIR.glob("*_model.joblib"))
        if joblib_files:
            rows = []
            for f in joblib_files:
                stat = f.stat()
                name = f.stem.replace("_model", "")
                m = model_metrics.get(name, {})
                rows.append({
                    "Model": name,
                    "Size": _fmt_bytes(stat.st_size),
                    "R²": m.get("r2", "—"),
                    "RMSE": m.get("rmse", "—"),
                    "MAE": m.get("mae", "—"),
                    "Accuracy": m.get("accuracy", "—"),
                    "F1": m.get("f1", "—"),
                    "Status": "LOADED",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if model_metrics:
                st.markdown(
                    '<p style="color:#6e7d8f; font-size:0.75rem; margin-top:8px;">'
                    "Detailed model performance comparison</p>",
                    unsafe_allow_html=True,
                )
                mr1, mr2, mr3, mr4 = st.columns(4)
                best_name = max(model_metrics, key=lambda k: model_metrics[k].get("r2", 0))
                best = model_metrics[best_name]
                with mr1:
                    st.markdown(
                        _metric_card("Best Model", best_name, "cyan"),
                        unsafe_allow_html=True,
                    )
                with mr2:
                    st.markdown(
                        _metric_card("Best R²", f"{best.get('r2', 0):.4f}", "green"),
                        unsafe_allow_html=True,
                    )
                with mr3:
                    st.markdown(
                        _metric_card("Best F1", f"{best.get('f1', 0):.4f}", "purple"),
                        unsafe_allow_html=True,
                    )
                with mr4:
                    st.markdown(
                        _metric_card("AUC (High-Risk)", f"{best.get('auc', 0):.4f}", "amber"),
                        unsafe_allow_html=True,
                    )

                with st.expander("Full Metrics Table"):
                    full_rows = []
                    for mname, mvals in model_metrics.items():
                        full_rows.append({
                            "Model": mname,
                            "RMSE": mvals.get("rmse"),
                            "MAE": mvals.get("mae"),
                            "R²": mvals.get("r2"),
                            "Correlation": mvals.get("correlation"),
                            "Accuracy": mvals.get("accuracy"),
                            "F1 (weighted)": mvals.get("f1"),
                            "Precision": mvals.get("precision"),
                            "Recall": mvals.get("recall"),
                            "High-Risk F1": mvals.get("high_risk_f1"),
                            "AUC": mvals.get("auc"),
                            "Train Time (s)": mvals.get("train_time_s"),
                            "Size (KB)": mvals.get("model_size_kb"),
                        })
                    st.dataframe(pd.DataFrame(full_rows), use_container_width=True, hide_index=True)
        else:
            st.markdown(
                _alert_card("No Models", "No model files found in models_saved/", "warning"),
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            _alert_card("Directory Missing", "models_saved/ does not exist — run the training pipeline", "warning"),
            unsafe_allow_html=True,
        )

    _divider()

    # ──────────────────────────────────────────────────────────────────
    # 5. Pipeline Artifacts
    # ──────────────────────────────────────────────────────────────────
    st.markdown("#### Pipeline Artifacts")
    st.markdown(
        '<p style="color:#6e7d8f; font-size:0.8rem;">Output files in proofs/</p>',
        unsafe_allow_html=True,
    )

    if PROOFS_DIR.exists():
        artifact_files = sorted(PROOFS_DIR.iterdir())
        artifact_files = [f for f in artifact_files if f.is_file()]
        if artifact_files:
            rows = []
            for f in artifact_files:
                stat = f.stat()
                highlight = f.name == "pipeline_results.json"
                rows.append({
                    "File": f.name,
                    "Size": _fmt_bytes(stat.st_size),
                    "Modified": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "Key": "PRIMARY" if highlight else "",
                })
            df_art = pd.DataFrame(rows)
            st.dataframe(df_art, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                _alert_card("No Artifacts", "proofs/ is empty — run the pipeline first", "warning"),
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            _alert_card("Directory Missing", "proofs/ does not exist", "warning"),
            unsafe_allow_html=True,
        )

    _divider()

    # ──────────────────────────────────────────────────────────────────
    # 6. System Performance Gauges
    # ──────────────────────────────────────────────────────────────────
    st.markdown("#### System Performance")

    # Compute values
    n_models = len(list(MODELS_DIR.glob("*.joblib"))) if MODELS_DIR.exists() else 0

    data_path = DATA_DIR / "train_data.csv"
    if data_path.exists():
        try:
            n_rows = sum(1 for _ in open(data_path, encoding="utf-8")) - 1
        except Exception:
            n_rows = 0
    else:
        n_rows = 0
    data_pct = min(100, (n_rows / 162_000) * 100) if n_rows > 0 else 0

    n_tests = _count_files(TESTS_DIR, "test_*.py") if TESTS_DIR.exists() else 0
    n_code_modules = _count_files(SRC_DIR, "*.py") if SRC_DIR.exists() else 0

    g1, g2, g3, g4 = st.columns(4)

    def _gauge(title, value, max_val, suffix="", color="#00D4FF"):
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=value,
                number=dict(suffix=suffix, font=dict(size=28, color="#E8EDF4")),
                title=dict(text=title, font=dict(size=13, color="#6e7d8f")),
                gauge=dict(
                    axis=dict(range=[0, max_val], tickcolor="#3d4f63", tickfont=dict(color="#3d4f63")),
                    bar=dict(color=color),
                    bgcolor="#111820",
                    bordercolor="#1a2332",
                    steps=[
                        dict(range=[0, max_val * 0.5], color="rgba(17,24,32,0.8)"),
                        dict(range=[max_val * 0.5, max_val], color="rgba(17,24,32,0.5)"),
                    ],
                ),
            )
        )
        fig.update_layout(**_PLOTLY_BASE, height=220)
        return fig

    with g1:
        st.plotly_chart(
            _gauge("Models Loaded", n_models, 6, color="#00D4FF"),
            use_container_width=True,
        )
    with g2:
        st.plotly_chart(
            _gauge("Data Coverage", round(data_pct, 1), 100, suffix="%", color="#00E87B"),
            use_container_width=True,
        )
    with g3:
        st.plotly_chart(
            _gauge("Test Files", n_tests, 20, color="#7B61FF"),
            use_container_width=True,
        )
    with g4:
        st.plotly_chart(
            _gauge("Code Modules", n_code_modules, 80, color="#FFAA00"),
            use_container_width=True,
        )

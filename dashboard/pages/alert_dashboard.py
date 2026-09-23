"""Alert Dashboard page for Orbital Sentinel."""

import datetime
import json
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROOFS_DIR = PROJECT_ROOT / "proofs"

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="#06090f",
    plot_bgcolor="#06090f",
    font=dict(color="#E8EDF4", family="Inter"),
    margin=dict(l=50, r=20, t=30, b=50),
)
_GRID = dict(gridcolor="#111820")

SEVERITY_COLORS = {
    "CRITICAL": "#FF2D55",
    "WARNING": "#FFAA00",
    "INFO": "#00D4FF",
}

SATELLITES = [
    "ISS (ZARYA)", "Sentinel-2A", "Starlink-1234", "CryoSat-2",
    "COSMOS 2251 DEB", "Envisat", "Starlink-3087", "NOAA-19",
    "Iridium 33 DEB", "Tiangong", "MetOp-B", "ALOS-2",
    "Sentinel-1A", "WorldView-3", "SPOT-7", "Terra",
]


def metric_card(label, value, color=""):
    color_class = f" {color}" if color else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value{color_class}">{value}</div>
    </div>
    """


def alert_card(title, detail, level="info"):
    return f"""
    <div class="alert-card {level}">
        <strong>{title}</strong><br>
        <span style="color: #6e7d8f; font-size: 0.85rem;">{detail}</span>
    </div>
    """


def _generate_demo_alerts():
    rng = np.random.RandomState(42)
    now = datetime.datetime.now(datetime.timezone.utc)
    alerts = []

    severity_weights = [0.2, 0.4, 0.4]
    severities = ["CRITICAL", "WARNING", "INFO"]
    statuses = ["ACTIVE", "ACKNOWLEDGED", "RESOLVED"]

    for i in range(22):
        sev = rng.choice(severities, p=severity_weights)
        hours_ago = rng.uniform(0, 168)
        ts = now - datetime.timedelta(hours=hours_ago)
        obj = rng.choice(SATELLITES)
        miss_km = rng.exponential(2.0) + 0.05
        risk = rng.uniform(-12, -2)

        if sev == "CRITICAL":
            risk = rng.uniform(-5, -1)
            miss_km = rng.uniform(0.05, 0.8)
            status = rng.choice(["ACTIVE", "ACKNOWLEDGED"], p=[0.6, 0.4])
        elif sev == "WARNING":
            risk = rng.uniform(-7, -4)
            miss_km = rng.uniform(0.3, 2.0)
            status = rng.choice(statuses, p=[0.3, 0.3, 0.4])
        else:
            risk = rng.uniform(-12, -6)
            miss_km = rng.uniform(1.0, 10.0)
            status = rng.choice(statuses, p=[0.1, 0.2, 0.7])

        desc_templates = {
            "CRITICAL": [
                f"High collision probability detected for {obj}",
                f"Urgent: {obj} close approach under 1 km",
                f"TCA imminent for {obj} — risk level exceeds threshold",
            ],
            "WARNING": [
                f"Elevated risk for {obj} conjunction event",
                f"Miss distance below 2 km for {obj}",
                f"Risk escalation detected in {obj} CDM sequence",
            ],
            "INFO": [
                f"New CDM received for {obj}",
                f"Conjunction event tracked for {obj}",
                f"Updated orbit determination for {obj}",
            ],
        }

        alerts.append({
            "id": f"ALT-{2024:04d}-{i+1:04d}",
            "timestamp": ts,
            "severity": sev,
            "event_id": f"EVT-2024-{rng.randint(1, 200):03d}",
            "object_name": obj,
            "miss_distance_km": round(miss_km, 3),
            "risk_score": round(risk, 2),
            "status": status,
            "description": rng.choice(desc_templates[sev]),
        })

    return sorted(alerts, key=lambda a: a["timestamp"], reverse=True)


def render_alert_dashboard():
    st.markdown(
        '<p style="color:#7a8899; font-size:0.85rem; margin-bottom:16px;">'
        'Real-time conjunction alert monitoring and notification management</p>',
        unsafe_allow_html=True,
    )

    alerts = _generate_demo_alerts()
    df = pd.DataFrame(alerts)

    # ── Summary metrics ──
    total = len(alerts)
    critical = sum(1 for a in alerts if a["severity"] == "CRITICAL")
    warnings = sum(1 for a in alerts if a["severity"] == "WARNING")
    active = sum(1 for a in alerts if a["status"] == "ACTIVE")
    resolved = sum(1 for a in alerts if a["status"] == "RESOLVED")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(metric_card("Total Alerts", str(total), "cyan"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Critical", str(critical), "red"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Warnings", str(warnings), "amber"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Active", str(active), "cyan"), unsafe_allow_html=True)
    with c5:
        st.markdown(metric_card("Resolved", str(resolved), "green"), unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Active alerts panel ──
    st.markdown("#### Active Alerts")
    active_alerts = [a for a in alerts if a["status"] == "ACTIVE"]
    active_alerts.sort(key=lambda a: {"CRITICAL": 0, "WARNING": 1, "INFO": 2}[a["severity"]])

    if not active_alerts:
        st.markdown(alert_card("All Clear", "No active alerts at this time", "info"), unsafe_allow_html=True)
    else:
        for a in active_alerts:
            sev_color = SEVERITY_COLORS[a["severity"]]
            dot_class = "critical" if a["severity"] == "CRITICAL" else "warning" if a["severity"] == "WARNING" else "active"
            ts_str = a["timestamp"].strftime("%Y-%m-%d %H:%M UTC")
            st.markdown(f"""
            <div class="alert-card {'critical' if a['severity'] == 'CRITICAL' else 'warning' if a['severity'] == 'WARNING' else 'info'}">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <div>
                        <span class="status-dot {dot_class}"></span>
                        <span style="color:{sev_color}; font-weight:700; font-size:0.75rem;
                                     font-family:'JetBrains Mono',monospace; letter-spacing:0.08em;">
                            {a['severity']}</span>
                        <span style="color:#3d4f63; margin:0 6px;">|</span>
                        <span style="color:#6e7d8f; font-size:0.8rem;">{a['event_id']}</span>
                    </div>
                    <span style="color:#3d4f63; font-size:0.7rem; font-family:'JetBrains Mono',monospace;">{ts_str}</span>
                </div>
                <strong style="color:#E8EDF4;">{a['object_name']}</strong>
                <span style="color:#6e7d8f;"> — {a['description']}</span>
                <div style="display:flex; gap:20px; margin-top:8px;">
                    <span style="font-size:0.75rem; color:#6e7d8f;">
                        Miss: <span style="color:#00D4FF; font-family:'JetBrains Mono',monospace;">{a['miss_distance_km']:.3f} km</span>
                    </span>
                    <span style="font-size:0.75rem; color:#6e7d8f;">
                        Risk: <span style="color:{sev_color}; font-family:'JetBrains Mono',monospace;">{a['risk_score']:.2f}</span>
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Alert timeline ──
    st.markdown("#### Alert Timeline")
    severity_y = {"CRITICAL": 3, "WARNING": 2, "INFO": 1}
    df["sev_y"] = df["severity"].map(severity_y)
    df["color"] = df["severity"].map(SEVERITY_COLORS)
    df["size"] = df["risk_score"].apply(lambda r: max(6, min(20, (r + 15) * 2)))

    fig_timeline = go.Figure()
    for sev in ["CRITICAL", "WARNING", "INFO"]:
        mask = df["severity"] == sev
        subset = df[mask]
        if subset.empty:
            continue
        fig_timeline.add_trace(go.Scatter(
            x=subset["timestamp"],
            y=subset["sev_y"],
            mode="markers",
            name=sev,
            marker=dict(
                color=SEVERITY_COLORS[sev],
                size=subset["size"],
                opacity=0.8,
                line=dict(width=1, color="rgba(255,255,255,0.15)"),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Object: %{customdata[1]}<br>"
                "Miss: %{customdata[2]:.3f} km<br>"
                "Risk: %{customdata[3]:.2f}<br>"
                "Time: %{x}<extra></extra>"
            ),
            customdata=subset[["event_id", "object_name", "miss_distance_km", "risk_score"]].values,
        ))

    fig_timeline.update_layout(
        **_PLOTLY_LAYOUT,
        height=320,
        yaxis=dict(tickvals=[1, 2, 3], ticktext=["INFO", "WARNING", "CRITICAL"], **_GRID),
        xaxis=dict(title="Time", **_GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Severity breakdown ──
    st.markdown("#### Severity Breakdown")
    sb1, sb2 = st.columns(2)

    with sb1:
        sev_counts = df["severity"].value_counts()
        fig_donut = go.Figure(go.Pie(
            labels=sev_counts.index.tolist(),
            values=sev_counts.values.tolist(),
            hole=0.55,
            marker=dict(colors=[SEVERITY_COLORS.get(s, "#6e7d8f") for s in sev_counts.index]),
            textfont=dict(color="#E8EDF4", size=12),
            hovertemplate="%{label}: %{value} alerts<extra></extra>",
        ))
        fig_donut.update_layout(
            **_PLOTLY_LAYOUT,
            height=300,
            showlegend=True,
            legend=dict(font=dict(color="#6e7d8f")),
            annotations=[dict(
                text=f"<b>{total}</b><br><span style='font-size:10px'>Total</span>",
                x=0.5, y=0.5, font=dict(size=18, color="#E8EDF4"), showarrow=False,
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with sb2:
        df["date"] = df["timestamp"].dt.date
        daily = df.groupby(["date", "severity"]).size().reset_index(name="count")
        fig_daily = go.Figure()
        for sev in ["CRITICAL", "WARNING", "INFO"]:
            sev_data = daily[daily["severity"] == sev]
            if sev_data.empty:
                continue
            fig_daily.add_trace(go.Bar(
                x=sev_data["date"],
                y=sev_data["count"],
                name=sev,
                marker_color=SEVERITY_COLORS[sev],
                opacity=0.85,
            ))
        fig_daily.update_layout(
            **_PLOTLY_LAYOUT,
            height=300,
            barmode="stack",
            xaxis=dict(title="Date", **_GRID),
            yaxis=dict(title="Alerts", **_GRID),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_daily, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Alert rules ──
    st.markdown("#### Alert Rules Configuration")
    with st.expander("View active alert rules", expanded=False):
        rules = [
            {"Rule": "High Risk Threshold", "Trigger": "log10(Pc) > -5",
             "Action": "CRITICAL alert", "Status": "ACTIVE",
             "Description": "Triggers when collision probability exceeds 1e-5"},
            {"Rule": "Close Approach", "Trigger": "miss_distance < 1 km",
             "Action": "WARNING alert", "Status": "ACTIVE",
             "Description": "Triggers when predicted miss distance falls below 1 km"},
            {"Rule": "Rapid Risk Escalation", "Trigger": "risk delta > 2 orders",
             "Action": "WARNING alert", "Status": "ACTIVE",
             "Description": "Triggers when risk increases by 2+ orders of magnitude between CDMs"},
            {"Rule": "TCA Approaching", "Trigger": "time_to_tca < 24h AND risk > MEDIUM",
             "Action": "CRITICAL alert", "Status": "ACTIVE",
             "Description": "Triggers when TCA is imminent and risk level is elevated"},
        ]
        rules_df = pd.DataFrame(rules)
        st.dataframe(rules_df, use_container_width=True, hide_index=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Alert history table ──
    st.markdown("#### Alert History")
    filter_sev = st.multiselect(
        "Filter by severity", ["CRITICAL", "WARNING", "INFO"],
        default=["CRITICAL", "WARNING", "INFO"], key="alert_sev_filter",
    )

    filtered = df[df["severity"].isin(filter_sev)].copy()
    filtered["timestamp"] = filtered["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    display_cols = ["id", "timestamp", "severity", "status", "object_name",
                    "event_id", "miss_distance_km", "risk_score", "description"]
    avail = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[avail].sort_values("timestamp", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=400,
    )

"""Orbital Tracker — real-time propagation and conjunction geometry analysis."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

R_EARTH = 6371.0  # km
MU_EARTH = 398600.4418  # km^3/s^2

_LAYOUT = dict(
    paper_bgcolor="#06090f",
    plot_bgcolor="#06090f",
    font=dict(color="#E8EDF4", family="Inter"),
    margin=dict(l=50, r=20, t=30, b=50),
)
_GRID = dict(gridcolor="#111820")

SATELLITE_COLORS = {
    "ISS (ZARYA)": "#00D4FF",
    "Sentinel-2A": "#FF2D55",
    "CryoSat-2": "#00E87B",
    "Starlink-4781": "#FFAA00",
    "COSMOS 2251 DEB": "#7B61FF",
    "Envisat": "#3B82F6",
}

CATALOG = {
    "ISS (ZARYA)": dict(a=6778.0, e=0.0002, i=51.6, raan=45.0, aop=0.0, M0=0.0),
    "Sentinel-2A": dict(a=7159.0, e=0.001, i=98.6, raan=120.0, aop=90.0, M0=30.0),
    "CryoSat-2": dict(a=7095.0, e=0.001, i=92.0, raan=200.0, aop=270.0, M0=60.0),
    "Starlink-4781": dict(a=6921.0, e=0.0001, i=53.0, raan=310.0, aop=0.0, M0=120.0),
    "COSMOS 2251 DEB": dict(a=7200.0, e=0.01, i=74.0, raan=80.0, aop=180.0, M0=200.0),
    "Envisat": dict(a=7159.0, e=0.001, i=98.5, raan=125.0, aop=85.0, M0=90.0),
}


# ---------------------------------------------------------------------------
# Orbital mechanics
# ---------------------------------------------------------------------------

def _solve_kepler(M: np.ndarray, e: float, tol: float = 1e-10) -> np.ndarray:
    """Solve Kepler's equation  M = E - e*sin(E)  via Newton-Raphson."""
    E = M.copy()
    for _ in range(50):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def _keplerian_to_eci(
    a: float, e: float, i_deg: float,
    raan_deg: float, aop_deg: float,
    true_anom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Keplerian elements + array of true anomalies to ECI (x, y, z) km."""
    i = np.radians(i_deg)
    raan = np.radians(raan_deg)
    aop = np.radians(aop_deg)
    nu = true_anom

    r_mag = a * (1.0 - e**2) / (1.0 + e * np.cos(nu))

    # Perifocal frame
    xp = r_mag * np.cos(nu)
    yp = r_mag * np.sin(nu)

    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_aop, sin_aop = np.cos(aop), np.sin(aop)
    cos_i, sin_i = np.cos(i), np.sin(i)

    # Rotation to ECI
    x = (cos_raan * cos_aop - sin_raan * sin_aop * cos_i) * xp + \
        (-cos_raan * sin_aop - sin_raan * cos_aop * cos_i) * yp
    y = (sin_raan * cos_aop + cos_raan * sin_aop * cos_i) * xp + \
        (-sin_raan * sin_aop + cos_raan * cos_aop * cos_i) * yp
    z = (sin_aop * sin_i) * xp + (cos_aop * sin_i) * yp

    return x, y, z


def _propagate(params: dict, t_seconds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate a satellite along its orbit for given time array."""
    a, e = params["a"], params["e"]
    n = np.sqrt(MU_EARTH / a**3)  # mean motion rad/s
    M = np.radians(params["M0"]) + n * t_seconds
    E = _solve_kepler(M, e)
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return _keplerian_to_eci(a, e, params["i"], params["raan"], params["aop"], nu)


def _orbital_period(a: float) -> float:
    return 2.0 * np.pi * np.sqrt(a**3 / MU_EARTH)


# ---------------------------------------------------------------------------
# Earth wireframe
# ---------------------------------------------------------------------------

def _earth_traces() -> list:
    """Return Plotly traces for a wireframe Earth sphere."""
    n_lon, n_lat = 36, 18
    phi = np.linspace(0, 2 * np.pi, n_lon)
    theta = np.linspace(0, np.pi, n_lat)

    traces = []
    wire_style = dict(color="rgba(0,212,255,0.12)", width=1)

    for t in theta:
        x = R_EARTH * np.sin(t) * np.cos(phi)
        y = R_EARTH * np.sin(t) * np.sin(phi)
        z = R_EARTH * np.cos(t) * np.ones_like(phi)
        traces.append(go.Scatter3d(
            x=x, y=y, z=z, mode="lines",
            line=wire_style, showlegend=False, hoverinfo="skip",
        ))

    for p in phi[::2]:
        x = R_EARTH * np.sin(theta) * np.cos(p)
        y = R_EARTH * np.sin(theta) * np.sin(p)
        z = R_EARTH * np.cos(theta)
        traces.append(go.Scatter3d(
            x=x, y=y, z=z, mode="lines",
            line=wire_style, showlegend=False, hoverinfo="skip",
        ))

    return traces


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def metric_card(label, value, color=""):
    color_class = f" {color}" if color else ""
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value{color_class}">{value}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_orbital_tracker():
    st.markdown(
        '<p style="color:#7a8899; font-size:0.85rem; margin-bottom:16px;">'
        'Real-time orbital propagation and conjunction geometry analysis</p>',
        unsafe_allow_html=True,
    )

    names = list(CATALOG.keys())

    # ── Controls ──
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        primary = st.selectbox("Primary object", names, index=0, key="ot_pri")
    with c2:
        sec_options = [n for n in names if n != primary]
        secondary = st.selectbox("Secondary object", sec_options, index=min(3, len(sec_options) - 1), key="ot_sec")
    with c3:
        hours = st.slider("Time window (hours)", 1, 24, 6, key="ot_hrs")
    with c4:
        n_frames = st.slider("Animation frames", 20, 60, 40, key="ot_frames")

    simulate = st.button("Run propagation", key="ot_sim", type="primary")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Propagation ──
    total_sec = hours * 3600.0
    t_orbit = np.linspace(0, total_sec, 600)
    t_frames = np.linspace(0, total_sec, n_frames)

    pri_params = CATALOG[primary]
    sec_params = CATALOG[secondary]

    xp, yp, zp = _propagate(pri_params, t_orbit)
    xs, ys, zs = _propagate(sec_params, t_orbit)

    xp_f, yp_f, zp_f = _propagate(pri_params, t_frames)
    xs_f, ys_f, zs_f = _propagate(sec_params, t_frames)

    dist_frames = np.sqrt((xp_f - xs_f)**2 + (yp_f - ys_f)**2 + (zp_f - zs_f)**2)
    min_dist = dist_frames.min()
    min_idx = int(dist_frames.argmin())
    tca_hours = t_frames[min_idx] / 3600.0

    # Relative velocity at each frame (finite difference)
    dt = t_frames[1] - t_frames[0] if len(t_frames) > 1 else 1.0
    vx_rel = np.gradient(xp_f - xs_f, dt)
    vy_rel = np.gradient(yp_f - ys_f, dt)
    vz_rel = np.gradient(zp_f - zs_f, dt)
    v_rel = np.sqrt(vx_rel**2 + vy_rel**2 + vz_rel**2)

    # ── Summary metrics ──
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(metric_card("Primary", primary, "cyan"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card("Secondary", secondary, "red"), unsafe_allow_html=True)
    with m3:
        color = "red" if min_dist < 10.0 else "amber" if min_dist < 100.0 else "green"
        st.markdown(metric_card("Min Distance", f"{min_dist:.2f} km", color), unsafe_allow_html=True)
    with m4:
        st.markdown(metric_card("TCA", f"T+{tca_hours:.2f} h", "purple"), unsafe_allow_html=True)
    with m5:
        st.markdown(metric_card("Rel Velocity", f"{v_rel[min_idx]:.2f} km/s", "amber"), unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 3D visualisation ──
    st.markdown("#### Conjunction Geometry")

    pri_color = SATELLITE_COLORS.get(primary, "#00D4FF")
    sec_color = SATELLITE_COLORS.get(secondary, "#FF2D55")

    fig = go.Figure()

    for tr in _earth_traces():
        fig.add_trace(tr)

    # Full orbit paths
    fig.add_trace(go.Scatter3d(
        x=xp, y=yp, z=zp, mode="lines",
        line=dict(color=pri_color, width=2),
        name=primary, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=sec_color, width=2),
        name=secondary, hoverinfo="skip",
    ))

    # Current positions (frame 0)
    fig.add_trace(go.Scatter3d(
        x=[xp_f[0]], y=[yp_f[0]], z=[zp_f[0]], mode="markers",
        marker=dict(size=7, color=pri_color, symbol="diamond"),
        name=f"{primary} position",
    ))
    fig.add_trace(go.Scatter3d(
        x=[xs_f[0]], y=[ys_f[0]], z=[zs_f[0]], mode="markers",
        marker=dict(size=7, color=sec_color, symbol="diamond"),
        name=f"{secondary} position",
    ))

    # TCA marker
    fig.add_trace(go.Scatter3d(
        x=[xp_f[min_idx]], y=[yp_f[min_idx]], z=[zp_f[min_idx]],
        mode="markers",
        marker=dict(size=10, color="#FFAA00", symbol="x", line=dict(width=2, color="#FFAA00")),
        name="TCA point",
    ))

    # Animation frames
    frames = []
    for k in range(n_frames):
        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=[xp_f[k]], y=[yp_f[k]], z=[zp_f[k]]),
                go.Scatter3d(x=[xs_f[k]], y=[ys_f[k]], z=[zs_f[k]]),
            ],
            traces=[len(fig.data) - 3, len(fig.data) - 2],
            name=str(k),
        ))
    fig.frames = frames

    # Camera aiming at midpoint of TCA
    cam_target = dict(
        x=float((xp_f[min_idx] + xs_f[min_idx]) / 2),
        y=float((yp_f[min_idx] + ys_f[min_idx]) / 2),
        z=float((zp_f[min_idx] + zs_f[min_idx]) / 2),
    )

    fig.update_layout(
        paper_bgcolor="#06090f",
        font=dict(color="#E8EDF4", family="Inter"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=620,
        showlegend=True,
        legend=dict(
            x=0.01, y=0.99, bgcolor="rgba(6,9,15,0.8)",
            bordercolor="#1a2332", borderwidth=1,
            font=dict(size=11),
        ),
        scene=dict(
            bgcolor="#06090f",
            xaxis=dict(title="X (km)", **_GRID, showbackground=False),
            yaxis=dict(title="Y (km)", **_GRID, showbackground=False),
            zaxis=dict(title="Z (km)", **_GRID, showbackground=False),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=0.8),
                center=dict(x=0, y=0, z=0),
            ),
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.02, y=0.02,
            xanchor="left", yanchor="bottom",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=80, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0),
                    )],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
            ],
            font=dict(color="#00D4FF"),
            bgcolor="rgba(0,212,255,0.08)",
            bordercolor="rgba(0,212,255,0.25)",
        )],
        sliders=[dict(
            active=0,
            steps=[dict(args=[[str(k)], dict(
                frame=dict(duration=80, redraw=True),
                mode="immediate",
                transition=dict(duration=0),
            )], label="", method="animate") for k in range(n_frames)],
            x=0.05, len=0.9,
            y=0, xanchor="left",
            currentvalue=dict(prefix="Frame: ", font=dict(size=11, color="#6e7d8f")),
            tickcolor="#1a2332",
            bordercolor="#1a2332",
            bgcolor="#0c1117",
            activebgcolor="#00D4FF",
            font=dict(color="#6e7d8f"),
        )],
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Conjunction geometry panels ──
    st.markdown("#### Approach Analysis")
    g1, g2 = st.columns(2)

    t_hours = t_frames / 3600.0

    with g1:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Scatter(
            x=t_hours, y=dist_frames, mode="lines",
            line=dict(color="#00D4FF", width=2),
            name="Miss distance",
            hovertemplate="T+%{x:.2f}h<br>%{y:.2f} km<extra></extra>",
        ))
        fig_dist.add_trace(go.Scatter(
            x=[t_hours[min_idx]], y=[min_dist], mode="markers",
            marker=dict(size=10, color="#FF2D55", symbol="x"),
            name=f"TCA ({min_dist:.2f} km)",
        ))
        fig_dist.update_layout(
            **_LAYOUT,
            height=320,
            xaxis=dict(title="Time (hours)", **_GRID),
            yaxis=dict(title="Distance (km)", **_GRID),
            showlegend=True,
            legend=dict(font=dict(size=10), bgcolor="rgba(6,9,15,0.8)"),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with g2:
        fig_vel = go.Figure()
        fig_vel.add_trace(go.Scatter(
            x=t_hours, y=v_rel, mode="lines",
            line=dict(color="#FFAA00", width=2),
            name="Relative velocity",
            hovertemplate="T+%{x:.2f}h<br>%{y:.2f} km/s<extra></extra>",
        ))
        fig_vel.update_layout(
            **_LAYOUT,
            height=320,
            xaxis=dict(title="Time (hours)", **_GRID),
            yaxis=dict(title="Velocity (km/s)", **_GRID),
            showlegend=True,
            legend=dict(font=dict(size=10), bgcolor="rgba(6,9,15,0.8)"),
        )
        st.plotly_chart(fig_vel, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── B-plane view ──
    st.markdown("#### B-Plane Projection")
    st.markdown(
        '<p style="color:#6e7d8f; font-size:0.8rem;">Closest-approach geometry projected '
        'onto the target B-plane (radial vs along-track miss components)</p>',
        unsafe_allow_html=True,
    )

    # Build a local RTN-like frame at TCA for the primary
    r_pri = np.array([xp_f[min_idx], yp_f[min_idx], zp_f[min_idx]])
    r_sec = np.array([xs_f[min_idx], ys_f[min_idx], zs_f[min_idx]])
    dr = r_sec - r_pri
    r_hat = r_pri / (np.linalg.norm(r_pri) + 1e-12)
    cross_track = dr - np.dot(dr, r_hat) * r_hat
    n_hat = np.cross(r_hat, np.array([0, 0, 1]))
    n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-12)

    # Project miss vectors at each frame onto B-plane
    b_r_vals = []
    b_t_vals = []
    for k in range(n_frames):
        rp = np.array([xp_f[k], yp_f[k], zp_f[k]])
        rs = np.array([xs_f[k], ys_f[k], zs_f[k]])
        delta = rs - rp
        b_r_vals.append(np.dot(delta, r_hat))
        b_t_vals.append(np.dot(delta, n_hat))

    fig_bp = go.Figure()
    fig_bp.add_trace(go.Scatter(
        x=b_t_vals, y=b_r_vals, mode="lines+markers",
        line=dict(color="#7B61FF", width=1.5),
        marker=dict(
            size=5, color=dist_frames,
            colorscale=[[0, "#FF2D55"], [0.5, "#FFAA00"], [1, "#00E87B"]],
            showscale=True,
            colorbar=dict(title="km", tickfont=dict(color="#6e7d8f"), titlefont=dict(color="#6e7d8f")),
        ),
        hovertemplate="B_T: %{x:.1f} km<br>B_R: %{y:.1f} km<extra></extra>",
        name="Approach trajectory",
    ))
    fig_bp.add_trace(go.Scatter(
        x=[b_t_vals[min_idx]], y=[b_r_vals[min_idx]], mode="markers",
        marker=dict(size=12, color="#FF2D55", symbol="x", line=dict(width=2, color="#FF2D55")),
        name="TCA",
    ))
    fig_bp.add_shape(
        type="circle", xref="x", yref="y",
        x0=-min_dist, y0=-min_dist, x1=min_dist, y1=min_dist,
        line=dict(color="rgba(255,45,85,0.3)", dash="dot"),
    )

    fig_bp.update_layout(
        **_LAYOUT,
        height=420,
        xaxis=dict(title="B-plane cross-track (km)", **_GRID, scaleanchor="y"),
        yaxis=dict(title="B-plane radial (km)", **_GRID),
        showlegend=True,
        legend=dict(font=dict(size=10), bgcolor="rgba(6,9,15,0.8)"),
    )
    st.plotly_chart(fig_bp, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Orbital elements table ──
    st.markdown("#### Orbital Parameters")
    oe1, oe2 = st.columns(2)
    for col, name, params, clr in [
        (oe1, primary, pri_params, "cyan"),
        (oe2, secondary, sec_params, "red"),
    ]:
        period = _orbital_period(params["a"])
        alt = params["a"] - R_EARTH
        with col:
            st.markdown(metric_card(name, f"{alt:.0f} km altitude", clr), unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:rgba(16,22,32,0.72); border:1px solid #1a2332;
                        border-radius:8px; padding:14px 18px; font-size:0.82rem;">
                <table style="width:100%; border-collapse:collapse;">
                    <tr><td style="color:#6e7d8f; padding:4px 0;">Semi-major axis</td>
                        <td style="color:#E8EDF4; text-align:right; font-family:'JetBrains Mono',monospace;">{params['a']:.1f} km</td></tr>
                    <tr><td style="color:#6e7d8f; padding:4px 0;">Eccentricity</td>
                        <td style="color:#E8EDF4; text-align:right; font-family:'JetBrains Mono',monospace;">{params['e']:.6f}</td></tr>
                    <tr><td style="color:#6e7d8f; padding:4px 0;">Inclination</td>
                        <td style="color:#E8EDF4; text-align:right; font-family:'JetBrains Mono',monospace;">{params['i']:.1f} deg</td></tr>
                    <tr><td style="color:#6e7d8f; padding:4px 0;">RAAN</td>
                        <td style="color:#E8EDF4; text-align:right; font-family:'JetBrains Mono',monospace;">{params['raan']:.1f} deg</td></tr>
                    <tr><td style="color:#6e7d8f; padding:4px 0;">Arg of perigee</td>
                        <td style="color:#E8EDF4; text-align:right; font-family:'JetBrains Mono',monospace;">{params['aop']:.1f} deg</td></tr>
                    <tr><td style="color:#6e7d8f; padding:4px 0;">Period</td>
                        <td style="color:#E8EDF4; text-align:right; font-family:'JetBrains Mono',monospace;">{period/60:.1f} min</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

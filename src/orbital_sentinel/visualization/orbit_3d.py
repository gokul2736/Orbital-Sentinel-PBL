"""3D orbit visualization using Plotly for conjunction analysis."""

import numpy as np
import plotly.graph_objects as go

R_EARTH_KM = 6378.137
MU_EARTH = 398600.4418


def kepler_orbit_points(
    sma_km: float,
    ecc: float,
    inc_deg: float,
    raan_deg: float = 0.0,
    argp_deg: float = 0.0,
    n_points: int = 360,
) -> np.ndarray:
    inc = np.radians(inc_deg)
    raan = np.radians(raan_deg)
    argp = np.radians(argp_deg)
    true_anomalies = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    p = sma_km * (1.0 - ecc ** 2)
    r = p / (1.0 + ecc * np.cos(true_anomalies))

    x_orb = r * np.cos(true_anomalies)
    y_orb = r * np.sin(true_anomalies)

    cos_argp, sin_argp = np.cos(argp), np.sin(argp)
    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_inc, sin_inc = np.cos(inc), np.sin(inc)

    x1 = x_orb * cos_argp - y_orb * sin_argp
    y1 = x_orb * sin_argp + y_orb * cos_argp

    x_eci = x1 * cos_raan - y1 * cos_inc * sin_raan
    y_eci = x1 * sin_raan + y1 * cos_inc * cos_raan
    z_eci = y1 * sin_inc

    return np.column_stack([x_eci, y_eci, z_eci])


def _orbit_point_at_anomaly(sma, ecc, inc_deg, raan_deg, argp_deg, nu_rad):
    inc = np.radians(inc_deg)
    raan = np.radians(raan_deg)
    argp = np.radians(argp_deg)
    p = sma * (1.0 - ecc ** 2)
    r = p / (1.0 + ecc * np.cos(nu_rad))
    x_o = r * np.cos(nu_rad)
    y_o = r * np.sin(nu_rad)
    x1 = x_o * np.cos(argp) - y_o * np.sin(argp)
    y1 = x_o * np.sin(argp) + y_o * np.cos(argp)
    x_eci = x1 * np.cos(raan) - y1 * np.cos(inc) * np.sin(raan)
    y_eci = x1 * np.sin(raan) + y1 * np.cos(inc) * np.cos(raan)
    z_eci = y1 * np.sin(inc)
    return np.array([x_eci, y_eci, z_eci])


def _derive_raan_argp(row, prefix):
    """Derive pseudo-realistic RAAN and ARGP from orbital & covariance data."""
    sma = row.get(f"{prefix}j2k_sma", 7000)
    inc = row.get(f"{prefix}j2k_inc", 50)
    ecc = row.get(f"{prefix}j2k_ecc", 0.001)
    sedr = abs(row.get(f"{prefix}sedr", 0))
    ct_r = row.get(f"{prefix}ct_r", 0)
    cn_r = row.get(f"{prefix}cn_r", 0)

    raan = ((sma * 0.051 + inc * 3.7 + sedr * 1e6) % 360.0)
    argp = ((ecc * 1e4 + abs(ct_r) * 0.01 + abs(cn_r) * 0.01 + inc * 1.3) % 360.0)
    return raan, argp


def create_earth_sphere(n_points: int = 80) -> tuple:
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = R_EARTH_KM * np.outer(np.cos(u), np.sin(v))
    y = R_EARTH_KM * np.outer(np.sin(u), np.sin(v))
    z = R_EARTH_KM * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def create_atmosphere_sphere(scale: float = 1.025, n_points: int = 40) -> tuple:
    """Slightly larger transparent sphere for atmospheric glow."""
    r = R_EARTH_KM * scale
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _earth_surface_color(n_points: int = 80) -> np.ndarray:
    """Generate a realistic latitude/longitude-based color map for Earth."""
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    lon, lat = np.meshgrid(u, v, indexing="ij")
    lat_deg = 90 - np.degrees(lat)
    lon_deg = np.degrees(lon)

    color = np.zeros_like(lat)

    ocean_mask = np.ones_like(lat, dtype=bool)

    continents = [
        ((-10, 30), (0.3, 1.8)),
        ((-10, 25), (3.5, 5.0)),
        ((20, 70), (0.0, 2.5)),
        ((35, 72), (0.8, 3.2)),
        ((-35, -15), (4.8, 5.8)),
        ((-55, -10), (5.0, 6.28)),
        ((10, 40), (1.8, 2.8)),
        ((25, 55), (1.0, 2.2)),
        ((-45, -10), (5.2, 6.0)),
        ((60, 85), (0.0, 6.28)),
        ((-90, -60), (0.0, 6.28)),
    ]
    for (lat_lo, lat_hi), (lon_lo, lon_hi) in continents:
        mask = (lat_deg >= lat_lo) & (lat_deg <= lat_hi) & (lon >= lon_lo) & (lon <= lon_hi)
        ocean_mask[mask] = False
        base = np.where(np.abs(lat_deg[mask]) < 25, 0.55, 0.65)
        color[mask] = base + 0.15 * np.sin(lat_deg[mask] * 0.04) + 0.05 * np.sin(lon_deg[mask] * 0.06)

    polar_mask = np.abs(lat_deg) > 65
    color[polar_mask & ~ocean_mask] = 0.88 + 0.07 * np.cos(lat_deg[polar_mask & ~ocean_mask] * 0.02)
    color[polar_mask & ocean_mask] = 0.82 + 0.05 * np.cos(lat_deg[polar_mask & ocean_mask] * 0.02)
    ocean_mask[polar_mask] = False

    depth = 0.12 + 0.08 * np.sin(lat_deg * 0.025 + 0.5) + 0.04 * np.sin(lon_deg * 0.03)
    color[ocean_mask] = depth[ocean_mask]

    return np.clip(color, 0.0, 1.0)


EARTH_COLORSCALE = [
    [0.00, "#050d1a"],
    [0.05, "#0a1a3a"],
    [0.10, "#0d2a5c"],
    [0.15, "#10367a"],
    [0.20, "#1a4a8a"],
    [0.25, "#1a5878"],
    [0.30, "#1a6050"],
    [0.35, "#1a6838"],
    [0.40, "#227530"],
    [0.45, "#2a802e"],
    [0.50, "#388a35"],
    [0.55, "#4a933a"],
    [0.60, "#5a9a40"],
    [0.65, "#6a9548"],
    [0.70, "#7a8d50"],
    [0.75, "#8a8050"],
    [0.80, "#9a7848"],
    [0.85, "#c0c8d0"],
    [0.90, "#d8dfe8"],
    [0.95, "#e8eef4"],
    [1.00, "#f4f8fc"],
]


def _risk_color(risk_value: float) -> str:
    if risk_value > -5:
        return "#FF3366"
    if risk_value > -7:
        return "#FFB800"
    if risk_value > -15:
        return "#00D4FF"
    return "#00FF88"


def _risk_label(risk_value: float) -> str:
    if risk_value > -5:
        return "HIGH"
    if risk_value > -7:
        return "MEDIUM"
    if risk_value > -15:
        return "LOW"
    return "NEGLIGIBLE"


ORBIT_COLORS = [
    "#00D4FF", "#FF6B9D", "#00FF88", "#FFB800",
    "#7C5CFC", "#FF3366", "#45E6B0", "#FFA94D",
]


def create_orbit_figure(
    events_df=None,
    selected_event_id=None,
    show_all_orbits: bool = True,
    dark_theme: bool = True,
) -> go.Figure:
    fig = go.Figure()

    n_pts = 90
    ex, ey, ez = create_earth_sphere(n_pts)
    surface_color = _earth_surface_color(n_pts)

    fig.add_trace(go.Surface(
        x=ex, y=ey, z=ez,
        surfacecolor=surface_color,
        colorscale=EARTH_COLORSCALE,
        showscale=False,
        opacity=0.97,
        name="Earth",
        hoverinfo="skip",
        lighting=dict(ambient=0.45, diffuse=0.65, specular=0.2, roughness=0.7, fresnel=0.15),
        lightposition=dict(x=15000, y=10000, z=12000),
    ))

    ax, ay, az = create_atmosphere_sphere(scale=1.02, n_points=40)
    fig.add_trace(go.Surface(
        x=ax, y=ay, z=az,
        surfacecolor=np.ones((40, 40)),
        colorscale=[[0, "rgba(60,160,255,0.0)"], [1, "rgba(60,160,255,0.06)"]],
        showscale=False,
        opacity=0.15,
        name="Atmosphere",
        hoverinfo="skip",
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
    ))

    if events_df is not None and len(events_df) > 0:
        if selected_event_id is not None:
            display_events = events_df[events_df["event_id"] == selected_event_id]
            if display_events.empty:
                display_events = events_df.head(5)
        elif show_all_orbits:
            unique_events = events_df["event_id"].unique()
            sample_ids = unique_events[:min(8, len(unique_events))]
            display_events = events_df[events_df["event_id"].isin(sample_ids)]
        else:
            display_events = events_df.head(1)

        plotted_orbits = set()
        conjunction_points = []

        for _, row in display_events.iterrows():
            eid = row.get("event_id", "?")
            row_dict = row.to_dict()

            t_sma = row.get("t_j2k_sma", 7000)
            t_ecc = min(row.get("t_j2k_ecc", 0.001), 0.9)
            t_inc = row.get("t_j2k_inc", 51.6)
            c_sma = row.get("c_j2k_sma", 7000)
            c_ecc = min(row.get("c_j2k_ecc", 0.01), 0.9)
            c_inc = row.get("c_j2k_inc", 98.0)

            risk = row.get("risk", -20.0)
            miss_dist = row.get("miss_distance", 0)

            t_raan, t_argp = _derive_raan_argp(row_dict, "t_")
            c_raan, c_argp = _derive_raan_argp(row_dict, "c_")

            t_key = f"t_{eid}"
            if t_key not in plotted_orbits and t_sma > 100:
                try:
                    pts = kepler_orbit_points(t_sma, t_ecc, t_inc, t_raan, t_argp, 500)
                    ci = len(plotted_orbits) % len(ORBIT_COLORS)
                    fig.add_trace(go.Scatter3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        mode="lines",
                        line=dict(color=ORBIT_COLORS[ci], width=2.5),
                        name=f"Target {eid}",
                        hoverinfo="name",
                        opacity=0.75,
                    ))
                    plotted_orbits.add(t_key)
                except (ValueError, ZeroDivisionError):
                    pass

            c_key = f"c_{eid}"
            if c_key not in plotted_orbits and c_sma > 100:
                try:
                    pts = kepler_orbit_points(c_sma, c_ecc, c_inc, c_raan, c_argp, 500)
                    ci = (len(plotted_orbits)) % len(ORBIT_COLORS)
                    fig.add_trace(go.Scatter3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        mode="lines",
                        line=dict(color=ORBIT_COLORS[ci], width=2, dash="dash"),
                        name=f"Chaser {eid}",
                        hoverinfo="name",
                        opacity=0.55,
                    ))
                    plotted_orbits.add(c_key)
                except (ValueError, ZeroDivisionError):
                    pass

            if t_sma > 100:
                nu = np.pi * 0.7
                try:
                    pos = _orbit_point_at_anomaly(t_sma, t_ecc, t_inc, t_raan, t_argp, nu)
                    conjunction_points.append({
                        "x": pos[0], "y": pos[1], "z": pos[2],
                        "event_id": eid, "risk": risk, "miss_distance": miss_dist,
                    })
                except (ValueError, ZeroDivisionError):
                    pass

        if conjunction_points:
            xs = [p["x"] for p in conjunction_points]
            ys = [p["y"] for p in conjunction_points]
            zs = [p["z"] for p in conjunction_points]
            colors = [_risk_color(p["risk"]) for p in conjunction_points]
            sizes = [max(5, min(16, 10 + p["risk"] * 0.5)) for p in conjunction_points]
            texts = [
                f"Event: {p['event_id']}<br>"
                f"Risk: {p['risk']:.1f} ({_risk_label(p['risk'])})<br>"
                f"Miss: {p['miss_distance']:,.0f} m"
                for p in conjunction_points
            ]

            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers",
                marker=dict(
                    size=sizes, color=colors, symbol="diamond",
                    opacity=0.95, line=dict(width=1, color="white"),
                ),
                text=texts, hoverinfo="text", name="Conjunctions",
            ))

            for i, p in enumerate(conjunction_points):
                if p["risk"] > -7:
                    fig.add_trace(go.Scatter3d(
                        x=[p["x"]], y=[p["y"]], z=[p["z"]],
                        mode="markers",
                        marker=dict(size=sizes[i] + 12, color=colors[i], opacity=0.12, line=dict(width=0)),
                        hoverinfo="skip", showlegend=False,
                    ))

    bg = "#0E1117" if dark_theme else "#FFFFFF"
    grid = "#151d2c" if dark_theme else "#E5E5E5"
    text = "#CDD6E0" if dark_theme else "#333333"

    axis_cfg = dict(
        range=[-12000, 12000],
        showbackground=True,
        backgroundcolor=bg,
        gridcolor=grid,
        showgrid=True,
        zeroline=False,
        showticklabels=False,
        title="",
    )

    fig.update_layout(
        scene=dict(
            xaxis=axis_cfg, yaxis=axis_cfg, zaxis=axis_cfg,
            aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=1.2, z=0.7), up=dict(x=0, y=0, z=1)),
        ),
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(color=text, family="Inter, sans-serif"),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            bgcolor="rgba(14,17,23,0.85)", bordercolor="#1a2332", borderwidth=1,
            font=dict(size=11, color=text), x=0.02, y=0.98,
        ),
        height=700,
    )

    return fig


def create_conjunction_detail_figure(
    row_dict: dict,
    physics_result: dict = None,
    dark_theme: bool = True,
) -> go.Figure:
    fig = go.Figure()

    t_sma = row_dict.get("t_j2k_sma", 7000)
    t_ecc = min(row_dict.get("t_j2k_ecc", 0.001), 0.9)
    t_inc = row_dict.get("t_j2k_inc", 51.6)
    c_sma = row_dict.get("c_j2k_sma", 7000)
    c_ecc = min(row_dict.get("c_j2k_ecc", 0.01), 0.9)
    c_inc = row_dict.get("c_j2k_inc", 98.0)

    t_raan, t_argp = _derive_raan_argp(row_dict, "t_")
    c_raan, c_argp = _derive_raan_argp(row_dict, "c_")

    try:
        t_pts = kepler_orbit_points(t_sma, t_ecc, t_inc, t_raan, t_argp, 600)
        fig.add_trace(go.Scatter3d(
            x=t_pts[:, 0], y=t_pts[:, 1], z=t_pts[:, 2],
            mode="lines", line=dict(color="#00D4FF", width=3),
            name="Target Orbit",
        ))
    except (ValueError, ZeroDivisionError):
        pass

    try:
        c_pts = kepler_orbit_points(c_sma, c_ecc, c_inc, c_raan, c_argp, 600)
        fig.add_trace(go.Scatter3d(
            x=c_pts[:, 0], y=c_pts[:, 1], z=c_pts[:, 2],
            mode="lines", line=dict(color="#FF3366", width=3, dash="dash"),
            name="Chaser Orbit",
        ))
    except (ValueError, ZeroDivisionError):
        pass

    nu = np.pi * 0.7
    try:
        t_pos = _orbit_point_at_anomaly(t_sma, t_ecc, t_inc, t_raan, t_argp, nu)

        miss_km = row_dict.get("miss_distance", 500) / 1000.0
        rel_r = row_dict.get("relative_position_r", 0)
        rel_t = row_dict.get("relative_position_t", 0)
        rel_n = row_dict.get("relative_position_n", 0)
        rel_vec = np.array([rel_r, rel_t, rel_n], dtype=float)
        rel_norm = np.linalg.norm(rel_vec)
        if rel_norm > 0:
            offset = rel_vec / rel_norm * max(miss_km, 30)
        else:
            offset = np.array([1.0, 0.5, 0.3]) / np.linalg.norm([1.0, 0.5, 0.3]) * max(miss_km, 30)
        c_pos = t_pos + offset

        fig.add_trace(go.Scatter3d(
            x=[t_pos[0]], y=[t_pos[1]], z=[t_pos[2]],
            mode="markers",
            marker=dict(size=10, color="#00D4FF", symbol="circle", line=dict(width=2, color="white")),
            name="Target @ TCA",
        ))
        fig.add_trace(go.Scatter3d(
            x=[c_pos[0]], y=[c_pos[1]], z=[c_pos[2]],
            mode="markers",
            marker=dict(size=10, color="#FF3366", symbol="circle", line=dict(width=2, color="white")),
            name="Chaser @ TCA",
        ))
        fig.add_trace(go.Scatter3d(
            x=[t_pos[0], c_pos[0]], y=[t_pos[1], c_pos[1]], z=[t_pos[2], c_pos[2]],
            mode="lines",
            line=dict(color="#FFB800", width=4, dash="dot"),
            name=f"Miss Vector ({miss_km:.1f} km)",
        ))

        sigma_r = max(row_dict.get("t_sigma_r", 100), 1)
        sigma_t = max(row_dict.get("t_sigma_t", 200), 1)
        sigma_n = max(row_dict.get("t_sigma_n", 100), 1)
        scale = 0.03
        u_e = np.linspace(0, 2 * np.pi, 25)
        v_e = np.linspace(0, np.pi, 15)
        ell_x = t_pos[0] + sigma_r * scale * np.outer(np.cos(u_e), np.sin(v_e))
        ell_y = t_pos[1] + sigma_t * scale * np.outer(np.sin(u_e), np.sin(v_e))
        ell_z = t_pos[2] + sigma_n * scale * np.outer(np.ones_like(u_e), np.cos(v_e))

        fig.add_trace(go.Surface(
            x=ell_x, y=ell_y, z=ell_z,
            colorscale=[[0, "rgba(0,212,255,0.12)"], [1, "rgba(0,212,255,0.12)"]],
            showscale=False, opacity=0.25,
            name="Covariance Ellipsoid", hoverinfo="skip",
        ))
    except (ValueError, ZeroDivisionError):
        pass

    n_detail = 80
    ex, ey, ez = create_earth_sphere(n_detail)
    surface_color = _earth_surface_color(n_detail)
    fig.add_trace(go.Surface(
        x=ex, y=ey, z=ez,
        surfacecolor=surface_color,
        colorscale=EARTH_COLORSCALE,
        showscale=False, opacity=0.95,
        hoverinfo="skip",
        lighting=dict(ambient=0.45, diffuse=0.65, specular=0.2, roughness=0.7, fresnel=0.15),
        lightposition=dict(x=15000, y=10000, z=12000),
    ))
    ax, ay, az = create_atmosphere_sphere(scale=1.02, n_points=36)
    fig.add_trace(go.Surface(
        x=ax, y=ay, z=az,
        surfacecolor=np.ones((36, 36)),
        colorscale=[[0, "rgba(60,160,255,0.0)"], [1, "rgba(60,160,255,0.06)"]],
        showscale=False, opacity=0.12,
        hoverinfo="skip", lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
    ))

    bg = "#0E1117" if dark_theme else "#FFFFFF"
    text = "#CDD6E0" if dark_theme else "#333333"
    grid = "#151d2c" if dark_theme else "#E5E5E5"

    axis_cfg = dict(
        showbackground=True, backgroundcolor=bg,
        gridcolor=grid, showgrid=True,
        zeroline=False, showticklabels=False, title="",
    )

    fig.update_layout(
        scene=dict(
            xaxis=axis_cfg, yaxis=axis_cfg, zaxis=axis_cfg,
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(color=text, family="Inter, sans-serif"),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            bgcolor="rgba(14,17,23,0.85)", bordercolor="#1a2332", borderwidth=1,
            font=dict(size=11, color=text),
        ),
        height=650,
    )

    return fig


def create_risk_timeline_figure(
    event_sequence: list,
    dark_theme: bool = True,
) -> go.Figure:
    if not event_sequence:
        fig = go.Figure()
        fig.add_annotation(text="No sequence data available", showarrow=False)
        return fig

    sorted_seq = sorted(event_sequence, key=lambda x: x.get("time_to_tca", 0), reverse=True)
    times = [s.get("time_to_tca", 0) for s in sorted_seq]
    risks = [s.get("risk", -30) for s in sorted_seq]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=times, y=risks,
        mode="lines+markers",
        name="Risk (log10 Pc)",
        line=dict(color="#00D4FF", width=3),
        marker=dict(
            size=10,
            color=[_risk_color(r) for r in risks],
            line=dict(width=2, color="white"),
        ),
        hovertemplate="Time to TCA: %{x:.2f} days<br>Risk: %{y:.2f}<extra></extra>",
    ))

    fig.add_hline(y=-5.0, line_dash="dash", line_color="#FF3366",
                  annotation_text="HIGH threshold", annotation_font_color="#FF3366")
    fig.add_hline(y=-7.0, line_dash="dash", line_color="#FFB800",
                  annotation_text="MEDIUM threshold", annotation_font_color="#FFB800")

    bg = "#0E1117" if dark_theme else "#FFFFFF"
    text = "#CDD6E0" if dark_theme else "#333333"
    grid = "#1a2332" if dark_theme else "#E5E5E5"

    fig.update_layout(
        xaxis=dict(title="Time to TCA (days)", autorange="reversed", gridcolor=grid, color=text),
        yaxis=dict(title="Risk — log10(Pc)", gridcolor=grid, color=text),
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(color=text, family="Inter, sans-serif"),
        margin=dict(l=60, r=20, t=20, b=60),
        legend=dict(bgcolor="rgba(14,17,23,0.8)", font=dict(color=text)),
        height=400,
    )

    return fig


def create_alert_matrix_figure(
    events_summary: list,
    dark_theme: bool = True,
) -> go.Figure:
    if not events_summary:
        fig = go.Figure()
        fig.add_annotation(text="No events to display", showarrow=False)
        return fig

    risks = [e.get("risk", -30) for e in events_summary]
    miss_dists = [e.get("miss_distance", 0) for e in events_summary]
    event_ids = [str(e.get("event_id", "?")) for e in events_summary]
    colors = [_risk_color(r) for r in risks]
    labels = [_risk_label(r) for r in risks]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=miss_dists, y=risks, mode="markers",
        marker=dict(size=10, color=colors, opacity=0.8, line=dict(width=1, color="white")),
        text=[f"Event: {eid}<br>Risk: {r:.1f} ({lbl})<br>Miss: {md:,.0f} m"
              for eid, r, lbl, md in zip(event_ids, risks, labels, miss_dists)],
        hoverinfo="text", name="Events",
    ))

    bg = "#0E1117" if dark_theme else "#FFFFFF"
    text = "#CDD6E0" if dark_theme else "#333333"
    grid = "#1a2332" if dark_theme else "#E5E5E5"

    fig.update_layout(
        xaxis=dict(title="Miss Distance (m)", type="log", gridcolor=grid, color=text),
        yaxis=dict(title="Risk — log10(Pc)", gridcolor=grid, color=text),
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(color=text, family="Inter, sans-serif"),
        margin=dict(l=60, r=20, t=20, b=60),
        height=450,
    )

    return fig


# =========================================================================
# Animated 3D CDM Simulation
# =========================================================================

def _solve_kepler_array(M_arr: np.ndarray, ecc: float, tol: float = 1e-10) -> np.ndarray:
    """Solve Kepler's equation M = E - e*sin(E) for an array of mean anomalies."""
    E = M_arr.copy()
    for _ in range(50):
        dE = (M_arr - E + ecc * np.sin(E)) / (1.0 - ecc * np.cos(E))
        E += dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def _eccentric_to_true(E: np.ndarray, ecc: float) -> np.ndarray:
    return 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(E / 2.0),
        np.sqrt(1.0 - ecc) * np.cos(E / 2.0),
    )


def _orbital_position_at_nu(
    sma: float, ecc: float, inc_deg: float,
    raan_deg: float, argp_deg: float, nu: np.ndarray,
) -> np.ndarray:
    """Compute ECI position(s) for given true anomaly array. Returns (N,3)."""
    inc = np.radians(inc_deg)
    raan = np.radians(raan_deg)
    argp = np.radians(argp_deg)

    p = sma * (1.0 - ecc ** 2)
    r = p / (1.0 + ecc * np.cos(nu))

    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    ca, sa = np.cos(argp), np.sin(argp)
    co, so = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)

    x1 = x_orb * ca - y_orb * sa
    y1 = x_orb * sa + y_orb * ca

    x_eci = x1 * co - y1 * ci * so
    y_eci = x1 * so + y1 * ci * co
    z_eci = y1 * si

    return np.column_stack([x_eci, y_eci, z_eci])


def _propagate_positions(
    sma: float, ecc: float, inc_deg: float,
    raan_deg: float, argp_deg: float, M_tca_deg: float,
    n_frames: int, span_seconds: float,
) -> np.ndarray:
    """Propagate satellite backward from TCA over span_seconds. Returns (n_frames, 3)."""
    ecc = min(ecc, 0.95)
    if sma <= 0:
        return np.zeros((n_frames, 3))

    mean_motion = np.sqrt(MU_EARTH / sma ** 3)
    M_tca = np.radians(M_tca_deg)

    dt = np.linspace(-span_seconds, 0.0, n_frames)
    M_arr = M_tca + mean_motion * dt
    M_arr = M_arr % (2.0 * np.pi)

    E_arr = _solve_kepler_array(M_arr, ecc)
    nu_arr = _eccentric_to_true(E_arr, ecc)

    return _orbital_position_at_nu(sma, ecc, inc_deg, raan_deg, argp_deg, nu_arr)


def _make_starfield(n_stars: int = 200, radius: float = 18000.0) -> go.Scatter3d:
    rng = np.random.RandomState(7)
    phi = rng.uniform(0, 2 * np.pi, n_stars)
    costheta = rng.uniform(-1, 1, n_stars)
    theta = np.arccos(costheta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    sizes = rng.uniform(0.8, 2.0, n_stars)
    return go.Scatter3d(
        x=x, y=y, z=z, mode="markers",
        marker=dict(size=sizes, color="white", opacity=0.4),
        hoverinfo="skip", showlegend=False,
    )


def _covariance_ellipsoid(
    center: np.ndarray,
    sigma_r: float, sigma_t: float, sigma_n: float,
    color: str, name: str, scale: float = 0.03,
    n_pts: int = 20, visible: bool = True,
) -> go.Surface:
    u = np.linspace(0, 2 * np.pi, n_pts)
    v = np.linspace(0, np.pi, n_pts)
    x = center[0] + sigma_r * scale * np.outer(np.cos(u), np.sin(v))
    y = center[1] + sigma_t * scale * np.outer(np.sin(u), np.sin(v))
    z = center[2] + sigma_n * scale * np.outer(np.ones_like(u), np.cos(v))
    rgba = color.lstrip("#")
    r_c, g_c, b_c = int(rgba[:2], 16), int(rgba[2:4], 16), int(rgba[4:6], 16)
    cs = [[0, f"rgba({r_c},{g_c},{b_c},0.15)"], [1, f"rgba({r_c},{g_c},{b_c},0.15)"]]
    return go.Surface(
        x=x, y=y, z=z, colorscale=cs,
        showscale=False, opacity=0.25,
        name=name, hoverinfo="skip",
        visible=visible,
    )


def create_cdm_3d_simulation(
    cdm_data: dict,
    sat1_name: str = "Target",
    sat2_name: str = "Chaser",
    n_frames: int = 60,
    dark_theme: bool = True,
) -> go.Figure:
    """Create animated 3D conjunction simulation from a CDM record.

    Renders Earth, both orbits, animated satellites approaching TCA,
    miss vector, covariance ellipsoids, and conjunction zone.
    """
    t_sma = max(float(cdm_data.get("t_j2k_sma", 7000)), 6400.0)
    t_ecc = min(float(cdm_data.get("t_j2k_ecc", 0.001)), 0.95)
    t_inc = float(cdm_data.get("t_j2k_inc", 51.6))
    t_raan = float(cdm_data.get("t_j2k_raan", 0.0))
    t_argp = float(cdm_data.get("t_j2k_argp", 0.0))
    t_M = float(cdm_data.get("t_j2k_mean_anomaly", 0.0))

    c_sma = max(float(cdm_data.get("c_j2k_sma", 7000)), 6400.0)
    c_ecc = min(float(cdm_data.get("c_j2k_ecc", 0.01)), 0.95)
    c_inc = float(cdm_data.get("c_j2k_inc", 98.0))
    c_raan = float(cdm_data.get("c_j2k_raan", 0.0))
    c_argp = float(cdm_data.get("c_j2k_argp", 0.0))
    c_M = float(cdm_data.get("c_j2k_mean_anomaly", 180.0))

    miss_dist_m = float(cdm_data.get("miss_distance", 500))
    rel_speed = float(cdm_data.get("relative_speed", 10000))
    risk = float(cdm_data.get("risk", -20.0)) if cdm_data.get("risk") is not None else -20.0

    t_period = 2.0 * np.pi * np.sqrt(t_sma ** 3 / MU_EARTH)
    span = min(t_period * 0.8, 3600.0)

    t_positions = _propagate_positions(
        t_sma, t_ecc, t_inc, t_raan, t_argp, t_M, n_frames, span,
    )
    c_positions = _propagate_positions(
        c_sma, c_ecc, c_inc, c_raan, c_argp, c_M, n_frames, span,
    )

    t_orbit = kepler_orbit_points(t_sma, t_ecc, t_inc, t_raan, t_argp, 500)
    c_orbit = kepler_orbit_points(c_sma, c_ecc, c_inc, c_raan, c_argp, 500)

    t_tca = t_positions[-1]
    c_tca = c_positions[-1]

    risk_col = _risk_color(risk)
    risk_lbl = _risk_label(risk)

    # -- Build figure: static traces first ----------------------------
    fig = go.Figure()

    # [0] star field
    fig.add_trace(_make_starfield())

    # [1] Earth
    n_anim = 90
    ex, ey, ez = create_earth_sphere(n_anim)
    surface_color = _earth_surface_color(n_anim)
    fig.add_trace(go.Surface(
        x=ex, y=ey, z=ez,
        surfacecolor=surface_color,
        colorscale=EARTH_COLORSCALE,
        showscale=False, opacity=0.97,
        name="Earth", hoverinfo="skip",
        lighting=dict(ambient=0.45, diffuse=0.65, specular=0.2, roughness=0.7, fresnel=0.15),
        lightposition=dict(x=15000, y=10000, z=12000),
    ))

    # [2] target orbit path
    fig.add_trace(go.Scatter3d(
        x=t_orbit[:, 0], y=t_orbit[:, 1], z=t_orbit[:, 2],
        mode="lines", line=dict(color="#00D4FF", width=2.5),
        name=f"{sat1_name} Orbit", opacity=0.5,
    ))
    # [3] chaser orbit path
    fig.add_trace(go.Scatter3d(
        x=c_orbit[:, 0], y=c_orbit[:, 1], z=c_orbit[:, 2],
        mode="lines", line=dict(color="#FF3366", width=2.5, dash="dash"),
        name=f"{sat2_name} Orbit", opacity=0.4,
    ))

    # Animated traces [4..14] ----------------------------------------
    trail_len = max(8, n_frames // 5)
    init_t_trail = t_positions[:trail_len]
    init_c_trail = c_positions[:trail_len]

    # [4] target trail
    fig.add_trace(go.Scatter3d(
        x=init_t_trail[:, 0], y=init_t_trail[:, 1], z=init_t_trail[:, 2],
        mode="lines", line=dict(color="#00D4FF", width=4),
        name=f"{sat1_name} Trail", showlegend=False,
    ))
    # [5] chaser trail
    fig.add_trace(go.Scatter3d(
        x=init_c_trail[:, 0], y=init_c_trail[:, 1], z=init_c_trail[:, 2],
        mode="lines", line=dict(color="#FF3366", width=4),
        name=f"{sat2_name} Trail", showlegend=False,
    ))
    # [6] target marker
    fig.add_trace(go.Scatter3d(
        x=[t_positions[0, 0]], y=[t_positions[0, 1]], z=[t_positions[0, 2]],
        mode="markers+text",
        marker=dict(size=8, color="#00D4FF", symbol="circle",
                    line=dict(width=2, color="white")),
        text=[sat1_name], textposition="top center",
        textfont=dict(size=10, color="#00D4FF"),
        name=sat1_name,
    ))
    # [7] target glow
    fig.add_trace(go.Scatter3d(
        x=[t_positions[0, 0]], y=[t_positions[0, 1]], z=[t_positions[0, 2]],
        mode="markers", marker=dict(size=18, color="#00D4FF", opacity=0.15),
        hoverinfo="skip", showlegend=False,
    ))
    # [8] chaser marker
    fig.add_trace(go.Scatter3d(
        x=[c_positions[0, 0]], y=[c_positions[0, 1]], z=[c_positions[0, 2]],
        mode="markers+text",
        marker=dict(size=8, color="#FF3366", symbol="circle",
                    line=dict(width=2, color="white")),
        text=[sat2_name], textposition="top center",
        textfont=dict(size=10, color="#FF3366"),
        name=sat2_name,
    ))
    # [9] chaser glow
    fig.add_trace(go.Scatter3d(
        x=[c_positions[0, 0]], y=[c_positions[0, 1]], z=[c_positions[0, 2]],
        mode="markers", marker=dict(size=18, color="#FF3366", opacity=0.15),
        hoverinfo="skip", showlegend=False,
    ))
    # [10] miss vector line (visible at TCA)
    fig.add_trace(go.Scatter3d(
        x=[t_tca[0], c_tca[0]], y=[t_tca[1], c_tca[1]], z=[t_tca[2], c_tca[2]],
        mode="lines", line=dict(color="#FFB800", width=5, dash="dot"),
        name=f"Miss Vector ({miss_dist_m:,.0f} m)",
        visible=False,
    ))
    # [11] conjunction glow
    mid = (t_tca + c_tca) / 2.0
    fig.add_trace(go.Scatter3d(
        x=[mid[0]], y=[mid[1]], z=[mid[2]],
        mode="markers",
        marker=dict(size=25, color=risk_col, opacity=0.12, line=dict(width=0)),
        hoverinfo="skip", showlegend=False,
        visible=False,
    ))
    # [12] conjunction marker
    fig.add_trace(go.Scatter3d(
        x=[mid[0]], y=[mid[1]], z=[mid[2]],
        mode="markers",
        marker=dict(size=10, color=risk_col, symbol="diamond",
                    line=dict(width=2, color="white")),
        text=[f"CONJUNCTION<br>Risk: {risk:.1f} ({risk_lbl})<br>"
              f"Miss: {miss_dist_m:,.0f} m<br>Rel Speed: {rel_speed:,.0f} m/s"],
        hoverinfo="text",
        name=f"TCA [{risk_lbl}]",
        visible=False,
    ))
    # [13] target covariance
    t_sig_r = max(float(cdm_data.get("t_sigma_r", 100)), 10.0)
    t_sig_t = max(float(cdm_data.get("t_sigma_t", 200)), 10.0)
    t_sig_n = max(float(cdm_data.get("t_sigma_n", 100)), 10.0)
    fig.add_trace(_covariance_ellipsoid(
        t_tca, t_sig_r, t_sig_t, t_sig_n, "#00D4FF",
        f"{sat1_name} Covariance", visible=False,
    ))
    # [14] chaser covariance
    c_sig_r = max(float(cdm_data.get("c_sigma_r", 100)), 10.0)
    c_sig_t = max(float(cdm_data.get("c_sigma_t", 200)), 10.0)
    c_sig_n = max(float(cdm_data.get("c_sigma_n", 100)), 10.0)
    fig.add_trace(_covariance_ellipsoid(
        c_tca, c_sig_r, c_sig_t, c_sig_n, "#FF3366",
        f"{sat2_name} Covariance", visible=False,
    ))

    # -- Animation frames ------------------------------------------
    animated_indices = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    frames = []
    time_labels = np.linspace(-span, 0.0, n_frames)

    for i in range(n_frames):
        trail_start = max(0, i - trail_len + 1)
        t_trail = t_positions[trail_start:i + 1]
        c_trail = c_positions[trail_start:i + 1]

        is_tca = i == n_frames - 1
        near_tca = i >= n_frames - 3

        t_sec = time_labels[i]
        if t_sec <= -60:
            lbl = f"T{t_sec/60.0:+.1f} min"
        else:
            lbl = f"T{t_sec:+.0f} s"

        frame_data = [
            go.Scatter3d(x=t_trail[:, 0], y=t_trail[:, 1], z=t_trail[:, 2]),
            go.Scatter3d(x=c_trail[:, 0], y=c_trail[:, 1], z=c_trail[:, 2]),

            go.Scatter3d(
                x=[t_positions[i, 0]], y=[t_positions[i, 1]], z=[t_positions[i, 2]],
                text=[sat1_name],
            ),
            go.Scatter3d(
                x=[t_positions[i, 0]], y=[t_positions[i, 1]], z=[t_positions[i, 2]],
                marker=dict(size=22 if near_tca else 18,
                            color="#00D4FF", opacity=0.2 if near_tca else 0.15),
            ),

            go.Scatter3d(
                x=[c_positions[i, 0]], y=[c_positions[i, 1]], z=[c_positions[i, 2]],
                text=[sat2_name],
            ),
            go.Scatter3d(
                x=[c_positions[i, 0]], y=[c_positions[i, 1]], z=[c_positions[i, 2]],
                marker=dict(size=22 if near_tca else 18,
                            color="#FF3366", opacity=0.2 if near_tca else 0.15),
            ),

            go.Scatter3d(visible=is_tca),
            go.Scatter3d(visible=is_tca),
            go.Scatter3d(visible=is_tca),
            go.Surface(visible=is_tca),
            go.Surface(visible=is_tca),
        ]

        frames.append(go.Frame(
            data=frame_data,
            traces=animated_indices,
            name=lbl,
        ))

    fig.frames = frames

    # -- Layout ---------------------------------------------------
    bg = "#0E1117" if dark_theme else "#FFFFFF"
    text_col = "#E0E0E0" if dark_theme else "#333333"
    grid_col = "#0a1628" if dark_theme else "#E5E5E5"

    axis_range = int(max(t_sma, c_sma) * 1.6)

    axis_cfg = dict(
        range=[-axis_range, axis_range],
        showbackground=True,
        backgroundcolor=bg,
        gridcolor=grid_col,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title="",
    )

    fig.update_layout(
        scene=dict(
            xaxis=axis_cfg, yaxis=axis_cfg, zaxis=axis_cfg,
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.5, y=1.2, z=0.6),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(color=text_col, family="Inter, sans-serif"),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            bgcolor="rgba(14, 17, 23, 0.85)",
            bordercolor="#1a2332",
            borderwidth=1,
            font=dict(size=11, color=text_col),
            x=0.01, y=0.99,
        ),
        height=750,
        title=dict(
            text=(f"<b>Conjunction Simulation</b> — {sat1_name} vs {sat2_name}"
                  f"  |  <span style='color:{risk_col}'>{risk_lbl} RISK</span>"),
            font=dict(size=14, color=text_col),
            x=0.5,
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.05, y=0.02,
            xanchor="left", yanchor="bottom",
            buttons=[
                dict(
                    label="  Play Approach  ",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=80, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=30, easing="cubic-in-out"),
                        mode="immediate",
                    )],
                ),
                dict(
                    label="  Pause  ",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                    )],
                ),
                dict(
                    label="  Jump to TCA  ",
                    method="animate",
                    args=[[frames[-1].name], dict(
                        frame=dict(duration=0, redraw=True),
                        mode="immediate",
                    )],
                ),
            ],
            font=dict(size=11, color="#E0E0E0"),
            bgcolor="rgba(28, 35, 51, 0.9)",
            bordercolor="#00D4FF",
            borderwidth=1,
        )],
        sliders=[dict(
            active=0,
            steps=[
                dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True),
                                          mode="immediate")],
                     label=f.name, method="animate")
                for f in frames[::max(1, n_frames // 20)]
            ],
            x=0.05, len=0.9,
            xanchor="left",
            y=-0.02,
            currentvalue=dict(
                prefix="Time: ", visible=True,
                font=dict(size=12, color=text_col),
            ),
            font=dict(size=9, color="#8B949E"),
            bgcolor="rgba(28, 35, 51, 0.8)",
            activebgcolor="#00D4FF",
            bordercolor="#1a2332",
            borderwidth=1,
            tickcolor="#8B949E",
        )],
    )

    return fig


def create_bplane_view(
    cdm_data: dict,
    dark_theme: bool = True,
) -> go.Figure:
    """Create a 2D B-plane view (radial vs cross-track miss geometry)."""
    miss_r = float(cdm_data.get("relative_position_r", 0))
    miss_n = float(cdm_data.get("relative_position_n", 0))
    miss_dist = float(cdm_data.get("miss_distance", 500)) / 1000.0

    t_sig_r = max(float(cdm_data.get("t_sigma_r", 100)), 1.0)
    t_sig_n = max(float(cdm_data.get("t_sigma_n", 100)), 1.0)
    c_sig_r = max(float(cdm_data.get("c_sigma_r", 100)), 1.0)
    c_sig_n = max(float(cdm_data.get("c_sigma_n", 100)), 1.0)

    risk = float(cdm_data.get("risk", -20.0)) if cdm_data.get("risk") is not None else -20.0
    risk_col = _risk_color(risk)

    fig = go.Figure()

    theta = np.linspace(0, 2 * np.pi, 100)
    for sigma_mult, opacity in [(1, 0.3), (2, 0.15), (3, 0.07)]:
        combined_r = np.sqrt(t_sig_r**2 + c_sig_r**2) * sigma_mult * 0.001
        combined_n = np.sqrt(t_sig_n**2 + c_sig_n**2) * sigma_mult * 0.001
        ell_x = combined_r * np.cos(theta)
        ell_y = combined_n * np.sin(theta)
        rgba = risk_col.lstrip("#")
        r_c, g_c, b_c = int(rgba[:2], 16), int(rgba[2:4], 16), int(rgba[4:6], 16)
        fig.add_trace(go.Scatter(
            x=ell_x, y=ell_y, mode="lines",
            line=dict(color=risk_col, width=1),
            fill="toself",
            fillcolor=f"rgba({r_c},{g_c},{b_c},{opacity})",
            name=f"{sigma_mult}-sigma",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(size=12, color="#00D4FF", symbol="circle",
                    line=dict(width=2, color="white")),
        name="Target (origin)",
    ))

    fig.add_trace(go.Scatter(
        x=[miss_r], y=[miss_n], mode="markers",
        marker=dict(size=12, color="#FF3366", symbol="diamond",
                    line=dict(width=2, color="white")),
        name=f"Chaser (miss={miss_dist:.3f} km)",
    ))

    fig.add_trace(go.Scatter(
        x=[0, miss_r], y=[0, miss_n], mode="lines",
        line=dict(color="#FFB800", width=3, dash="dot"),
        name="Miss Vector",
    ))

    bg = "#0E1117" if dark_theme else "#FFFFFF"
    text_col = "#E0E0E0" if dark_theme else "#333333"
    grid_col = "#1a2332" if dark_theme else "#E5E5E5"

    fig.update_layout(
        xaxis=dict(title="Radial (km)", gridcolor=grid_col, color=text_col,
                   scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Cross-Track (km)", gridcolor=grid_col, color=text_col),
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(color=text_col, family="Inter, sans-serif"),
        margin=dict(l=60, r=20, t=40, b=60),
        height=450,
        title=dict(
            text="<b>B-Plane View</b> — Close Approach Geometry",
            font=dict(size=13, color=text_col), x=0.5,
        ),
        legend=dict(
            bgcolor="rgba(14, 17, 23, 0.85)",
            bordercolor="#1a2332", borderwidth=1,
            font=dict(size=10, color=text_col),
        ),
    )

    return fig

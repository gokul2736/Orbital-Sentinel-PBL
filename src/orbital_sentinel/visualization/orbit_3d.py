"""3D orbit visualization using Plotly for conjunction analysis."""

import numpy as np
import plotly.graph_objects as go

R_EARTH_KM = 6378.137


def kepler_orbit_points(
    sma_km: float,
    ecc: float,
    inc_deg: float,
    raan_deg: float = 0.0,
    argp_deg: float = 0.0,
    n_points: int = 360,
) -> np.ndarray:
    """Compute 3D ECI coordinates for a Keplerian orbit.

    Returns array of shape (n_points, 3) in km.
    """
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


def create_earth_sphere(n_points: int = 50) -> tuple:
    """Generate Earth sphere mesh coordinates."""
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = R_EARTH_KM * np.outer(np.cos(u), np.sin(v))
    y = R_EARTH_KM * np.outer(np.sin(u), np.sin(v))
    z = R_EARTH_KM * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _earth_colorscale():
    return [
        [0.0, "#0a1628"],
        [0.15, "#0d2137"],
        [0.3, "#1a3a5c"],
        [0.45, "#1a5c3a"],
        [0.55, "#2d7a4f"],
        [0.7, "#1a5c3a"],
        [0.85, "#1a3a5c"],
        [1.0, "#0a1628"],
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


def create_orbit_figure(
    events_df=None,
    selected_event_id=None,
    show_all_orbits: bool = True,
    dark_theme: bool = True,
) -> go.Figure:
    """Create interactive 3D orbit visualization with Earth and conjunction events."""
    fig = go.Figure()

    ex, ey, ez = create_earth_sphere(60)
    np.random.seed(42)
    color_data = np.random.rand(*ex.shape)

    fig.add_trace(go.Surface(
        x=ex, y=ey, z=ez,
        surfacecolor=color_data,
        colorscale=_earth_colorscale(),
        showscale=False,
        opacity=0.95,
        name="Earth",
        hoverinfo="skip",
        lighting=dict(ambient=0.6, diffuse=0.5, specular=0.2),
    ))

    orbit_colors = [
        "#00D4FF", "#FF3366", "#00FF88", "#FFB800",
        "#7C5CFC", "#FF6B9D", "#45E6B0", "#FFA94D",
    ]

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

            t_sma = row.get("t_j2k_sma", 7000)
            t_ecc = row.get("t_j2k_ecc", 0.001)
            t_inc = row.get("t_j2k_inc", 51.6)
            c_sma = row.get("c_j2k_sma", 7000)
            c_ecc = row.get("c_j2k_ecc", 0.01)
            c_inc = row.get("c_j2k_inc", 98.0)

            risk = row.get("risk", -20.0)
            miss_dist = row.get("miss_distance", 0)

            t_key = f"t_{eid}"
            if t_key not in plotted_orbits and t_sma > 0:
                raan_t = (hash(str(eid) + "t") % 360)
                argp_t = (hash(str(eid) + "t_argp") % 360)
                try:
                    pts = kepler_orbit_points(t_sma, min(t_ecc, 0.95), t_inc, raan_t, argp_t)
                    ci = hash(str(eid)) % len(orbit_colors)
                    fig.add_trace(go.Scatter3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        mode="lines",
                        line=dict(color=orbit_colors[ci], width=2.5),
                        name=f"Target {eid}",
                        hoverinfo="name",
                        opacity=0.7,
                    ))
                    plotted_orbits.add(t_key)
                except (ValueError, ZeroDivisionError):
                    pass

            c_key = f"c_{eid}"
            if c_key not in plotted_orbits and c_sma > 0:
                raan_c = (hash(str(eid) + "c") % 360)
                argp_c = (hash(str(eid) + "c_argp") % 360)
                try:
                    pts = kepler_orbit_points(c_sma, min(c_ecc, 0.95), c_inc, raan_c, argp_c)
                    ci = (hash(str(eid)) + 1) % len(orbit_colors)
                    fig.add_trace(go.Scatter3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        mode="lines",
                        line=dict(color=orbit_colors[ci], width=2.5, dash="dash"),
                        name=f"Chaser {eid}",
                        hoverinfo="name",
                        opacity=0.5,
                    ))
                    plotted_orbits.add(c_key)
                except (ValueError, ZeroDivisionError):
                    pass

            if t_sma > 0:
                raan_t = (hash(str(eid) + "t") % 360)
                argp_t = (hash(str(eid) + "t_argp") % 360)
                nu = (hash(str(eid) + "conj") % 314) / 100.0
                try:
                    p = t_sma * (1.0 - min(t_ecc, 0.95) ** 2)
                    r = p / (1.0 + min(t_ecc, 0.95) * np.cos(nu))
                    inc = np.radians(t_inc)
                    raan = np.radians(raan_t)
                    argp = np.radians(argp_t)
                    x_o = r * np.cos(nu)
                    y_o = r * np.sin(nu)
                    x1 = x_o * np.cos(argp) - y_o * np.sin(argp)
                    y1 = x_o * np.sin(argp) + y_o * np.cos(argp)
                    cx = x1 * np.cos(raan) - y1 * np.cos(inc) * np.sin(raan)
                    cy = x1 * np.sin(raan) + y1 * np.cos(inc) * np.cos(raan)
                    cz = y1 * np.sin(inc)

                    conjunction_points.append({
                        "x": cx, "y": cy, "z": cz,
                        "event_id": eid,
                        "risk": risk,
                        "miss_distance": miss_dist,
                    })
                except (ValueError, ZeroDivisionError):
                    pass

        if conjunction_points:
            xs = [p["x"] for p in conjunction_points]
            ys = [p["y"] for p in conjunction_points]
            zs = [p["z"] for p in conjunction_points]
            colors = [_risk_color(p["risk"]) for p in conjunction_points]
            sizes = [max(6, min(20, 12 + p["risk"])) for p in conjunction_points]
            texts = [
                f"Event: {p['event_id']}<br>"
                f"Risk: {p['risk']:.1f} ({_risk_label(p['risk'])})<br>"
                f"Miss Distance: {p['miss_distance']:,.0f} m"
                for p in conjunction_points
            ]

            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=colors,
                    symbol="diamond",
                    opacity=0.95,
                    line=dict(width=1, color="white"),
                ),
                text=texts,
                hoverinfo="text",
                name="Conjunctions",
            ))

            for i, p in enumerate(conjunction_points):
                if p["risk"] > -7:
                    fig.add_trace(go.Scatter3d(
                        x=[p["x"]], y=[p["y"]], z=[p["z"]],
                        mode="markers",
                        marker=dict(
                            size=sizes[i] + 10,
                            color=colors[i],
                            opacity=0.15,
                            line=dict(width=0),
                        ),
                        hoverinfo="skip",
                        showlegend=False,
                    ))

    bg_color = "#0E1117" if dark_theme else "#FFFFFF"
    grid_color = "#1a2332" if dark_theme else "#E5E5E5"
    text_color = "#E0E0E0" if dark_theme else "#333333"

    axis_range = 12000

    axis_settings = dict(
        range=[-axis_range, axis_range],
        showbackground=True,
        backgroundcolor=bg_color,
        gridcolor=grid_color,
        showgrid=True,
        zeroline=False,
        showticklabels=False,
        title="",
    )

    fig.update_layout(
        scene=dict(
            xaxis=axis_settings,
            yaxis=axis_settings,
            zaxis=axis_settings,
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, family="Inter, sans-serif"),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            bgcolor="rgba(14, 17, 23, 0.8)",
            bordercolor="#1a2332",
            borderwidth=1,
            font=dict(size=11, color=text_color),
            x=0.02, y=0.98,
        ),
        height=700,
    )

    return fig


def create_conjunction_detail_figure(
    row_dict: dict,
    physics_result: dict = None,
    dark_theme: bool = True,
) -> go.Figure:
    """Create detailed 3D view of a single conjunction event with miss vector."""
    fig = go.Figure()

    t_sma = row_dict.get("t_j2k_sma", 7000)
    t_ecc = row_dict.get("t_j2k_ecc", 0.001)
    t_inc = row_dict.get("t_j2k_inc", 51.6)
    c_sma = row_dict.get("c_j2k_sma", 7000)
    c_ecc = row_dict.get("c_j2k_ecc", 0.01)
    c_inc = row_dict.get("c_j2k_inc", 98.0)

    eid = row_dict.get("event_id", 0)
    raan_t = (hash(str(eid) + "t") % 360)
    argp_t = (hash(str(eid) + "t_argp") % 360)
    raan_c = (hash(str(eid) + "c") % 360)
    argp_c = (hash(str(eid) + "c_argp") % 360)

    try:
        t_pts = kepler_orbit_points(t_sma, min(t_ecc, 0.95), t_inc, raan_t, argp_t, 500)
        fig.add_trace(go.Scatter3d(
            x=t_pts[:, 0], y=t_pts[:, 1], z=t_pts[:, 2],
            mode="lines",
            line=dict(color="#00D4FF", width=3),
            name="Target Orbit",
        ))
    except (ValueError, ZeroDivisionError):
        pass

    try:
        c_pts = kepler_orbit_points(c_sma, min(c_ecc, 0.95), c_inc, raan_c, argp_c, 500)
        fig.add_trace(go.Scatter3d(
            x=c_pts[:, 0], y=c_pts[:, 1], z=c_pts[:, 2],
            mode="lines",
            line=dict(color="#FF3366", width=3, dash="dash"),
            name="Chaser Orbit",
        ))
    except (ValueError, ZeroDivisionError):
        pass

    nu = 1.5
    try:
        p_t = t_sma * (1.0 - min(t_ecc, 0.95) ** 2)
        r_t = p_t / (1.0 + min(t_ecc, 0.95) * np.cos(nu))
        inc_r = np.radians(t_inc)
        raan_r = np.radians(raan_t)
        argp_r = np.radians(argp_t)
        x_o = r_t * np.cos(nu)
        y_o = r_t * np.sin(nu)
        x1 = x_o * np.cos(argp_r) - y_o * np.sin(argp_r)
        y1 = x_o * np.sin(argp_r) + y_o * np.cos(argp_r)
        t_x = x1 * np.cos(raan_r) - y1 * np.cos(inc_r) * np.sin(raan_r)
        t_y = x1 * np.sin(raan_r) + y1 * np.cos(inc_r) * np.cos(raan_r)
        t_z = y1 * np.sin(inc_r)

        miss_km = row_dict.get("miss_distance", 500) / 1000.0
        offset_dir = np.array([1.0, 0.5, 0.3])
        offset_dir = offset_dir / np.linalg.norm(offset_dir) * max(miss_km, 50)
        c_x = t_x + offset_dir[0]
        c_y = t_y + offset_dir[1]
        c_z = t_z + offset_dir[2]

        fig.add_trace(go.Scatter3d(
            x=[t_x], y=[t_y], z=[t_z],
            mode="markers",
            marker=dict(size=10, color="#00D4FF", symbol="circle",
                        line=dict(width=2, color="white")),
            name="Target Position",
        ))
        fig.add_trace(go.Scatter3d(
            x=[c_x], y=[c_y], z=[c_z],
            mode="markers",
            marker=dict(size=10, color="#FF3366", symbol="circle",
                        line=dict(width=2, color="white")),
            name="Chaser Position",
        ))

        fig.add_trace(go.Scatter3d(
            x=[t_x, c_x], y=[t_y, c_y], z=[t_z, c_z],
            mode="lines",
            line=dict(color="#FFB800", width=4, dash="dot"),
            name=f"Miss Distance ({miss_km:.1f} km)",
        ))

        sigma_r = row_dict.get("t_sigma_r", 100)
        sigma_t = row_dict.get("t_sigma_t", 200)
        sigma_n = row_dict.get("t_sigma_n", 100)
        scale = 0.05

        u_e = np.linspace(0, 2 * np.pi, 30)
        v_e = np.linspace(0, np.pi, 20)
        ell_x = t_x + sigma_r * scale * np.outer(np.cos(u_e), np.sin(v_e))
        ell_y = t_y + sigma_t * scale * np.outer(np.sin(u_e), np.sin(v_e))
        ell_z = t_z + sigma_n * scale * np.outer(np.ones_like(u_e), np.cos(v_e))

        fig.add_trace(go.Surface(
            x=ell_x, y=ell_y, z=ell_z,
            colorscale=[[0, "rgba(0,212,255,0.15)"], [1, "rgba(0,212,255,0.15)"]],
            showscale=False, opacity=0.3,
            name="Target Covariance",
            hoverinfo="skip",
        ))
    except (ValueError, ZeroDivisionError):
        pass

    ex, ey, ez = create_earth_sphere(40)
    np.random.seed(42)
    fig.add_trace(go.Surface(
        x=ex, y=ey, z=ez,
        surfacecolor=np.random.rand(*ex.shape),
        colorscale=_earth_colorscale(),
        showscale=False, opacity=0.9,
        hoverinfo="skip",
    ))

    bg_color = "#0E1117" if dark_theme else "#FFFFFF"
    text_color = "#E0E0E0" if dark_theme else "#333333"
    grid_color = "#1a2332" if dark_theme else "#E5E5E5"

    axis_settings = dict(
        showbackground=True,
        backgroundcolor=bg_color,
        gridcolor=grid_color,
        showgrid=True,
        zeroline=False,
        showticklabels=False,
        title="",
    )

    fig.update_layout(
        scene=dict(
            xaxis=axis_settings,
            yaxis=axis_settings,
            zaxis=axis_settings,
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, family="Inter, sans-serif"),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            bgcolor="rgba(14, 17, 23, 0.8)",
            bordercolor="#1a2332",
            borderwidth=1,
            font=dict(size=11, color=text_color),
        ),
        height=600,
    )

    return fig


def create_risk_timeline_figure(
    event_sequence: list,
    dark_theme: bool = True,
) -> go.Figure:
    """Create risk evolution timeline for a conjunction event."""
    if not event_sequence:
        fig = go.Figure()
        fig.add_annotation(text="No sequence data available", showarrow=False)
        return fig

    sorted_seq = sorted(event_sequence, key=lambda x: x.get("time_to_tca", 0), reverse=True)
    times = [s.get("time_to_tca", 0) for s in sorted_seq]
    risks = [s.get("risk", -30) for s in sorted_seq]
    miss_dists = [s.get("miss_distance", 0) for s in sorted_seq]

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
        hovertemplate=(
            "Time to TCA: %{x:.2f} days<br>"
            "Risk: %{y:.2f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.add_hline(y=-5.0, line_dash="dash", line_color="#FF3366",
                  annotation_text="HIGH threshold", annotation_font_color="#FF3366")
    fig.add_hline(y=-7.0, line_dash="dash", line_color="#FFB800",
                  annotation_text="MEDIUM threshold", annotation_font_color="#FFB800")

    fig.add_vrect(x0=min(times), x1=max(times), y0=-5.0, y1=max(risks) + 2,
                  fillcolor="rgba(255,51,102,0.05)", line_width=0)
    fig.add_vrect(x0=min(times), x1=max(times), y0=-7.0, y1=-5.0,
                  fillcolor="rgba(255,184,0,0.05)", line_width=0)

    bg_color = "#0E1117" if dark_theme else "#FFFFFF"
    text_color = "#E0E0E0" if dark_theme else "#333333"
    grid_color = "#1a2332" if dark_theme else "#E5E5E5"

    fig.update_layout(
        xaxis=dict(
            title="Time to TCA (days)",
            autorange="reversed",
            gridcolor=grid_color,
            color=text_color,
        ),
        yaxis=dict(
            title="Risk — log10(Pc)",
            gridcolor=grid_color,
            color=text_color,
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, family="Inter, sans-serif"),
        margin=dict(l=60, r=20, t=20, b=60),
        legend=dict(
            bgcolor="rgba(14, 17, 23, 0.8)",
            font=dict(color=text_color),
        ),
        height=400,
    )

    return fig


def create_alert_matrix_figure(
    events_summary: list,
    dark_theme: bool = True,
) -> go.Figure:
    """Create a risk matrix scatter of events: miss distance vs risk."""
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
        x=miss_dists, y=risks,
        mode="markers",
        marker=dict(
            size=10,
            color=colors,
            opacity=0.8,
            line=dict(width=1, color="white"),
        ),
        text=[f"Event: {eid}<br>Risk: {r:.1f} ({lbl})<br>Miss: {md:,.0f} m"
              for eid, r, lbl, md in zip(event_ids, risks, labels, miss_dists)],
        hoverinfo="text",
        name="Events",
    ))

    bg_color = "#0E1117" if dark_theme else "#FFFFFF"
    text_color = "#E0E0E0" if dark_theme else "#333333"
    grid_color = "#1a2332" if dark_theme else "#E5E5E5"

    fig.update_layout(
        xaxis=dict(title="Miss Distance (m)", type="log", gridcolor=grid_color, color=text_color),
        yaxis=dict(title="Risk — log10(Pc)", gridcolor=grid_color, color=text_color),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, family="Inter, sans-serif"),
        margin=dict(l=60, r=20, t=20, b=60),
        height=450,
    )

    return fig

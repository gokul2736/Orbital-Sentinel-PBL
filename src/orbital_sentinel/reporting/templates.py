"""Shared HTML/CSS templates for reports."""

from datetime import datetime, timezone

BASE_CSS = """
body {
    background: #0a0e14;
    color: #E6EDF3;
    font-family: 'Segoe UI', -apple-system, sans-serif;
    margin: 0;
    padding: 24px;
    line-height: 1.6;
}
.container { max-width: 1100px; margin: 0 auto; }
h1 {
    font-size: 1.6rem;
    font-weight: 700;
    border-bottom: 2px solid #1e2a3a;
    padding-bottom: 12px;
    margin-bottom: 24px;
}
h2 {
    font-size: 1.2rem;
    font-weight: 600;
    color: #00D4FF;
    margin-top: 32px;
    margin-bottom: 12px;
}
.header-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1e2a3a;
    padding-bottom: 16px;
    margin-bottom: 24px;
}
.header-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a5568;
}
.header-time {
    font-size: 0.75rem;
    color: #4a5568;
    font-family: monospace;
}
.metric-row {
    display: flex;
    gap: 16px;
    margin-bottom: 16px;
    flex-wrap: wrap;
}
.metric-card {
    background: rgba(21,28,40,0.65);
    border: 1px solid #1e2a3a;
    border-radius: 10px;
    padding: 14px 18px;
    flex: 1;
    min-width: 140px;
}
.metric-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4a5568;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    font-family: monospace;
}
.cyan { color: #00D4FF; }
.red { color: #FF3366; }
.amber { color: #FFB800; }
.green { color: #00FF88; }
.purple { color: #7C5CFC; }
.risk-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: monospace;
}
.risk-high { background: rgba(255,51,102,0.12); color: #FF3366; border: 1px solid rgba(255,51,102,0.25); }
.risk-medium { background: rgba(255,184,0,0.12); color: #FFB800; border: 1px solid rgba(255,184,0,0.25); }
.risk-low { background: rgba(0,212,255,0.12); color: #00D4FF; border: 1px solid rgba(0,212,255,0.25); }
.risk-negligible { background: rgba(0,255,136,0.12); color: #00FF88; border: 1px solid rgba(0,255,136,0.25); }
.alert-card {
    background: rgba(21,28,40,0.65);
    border-left: 3px solid;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.alert-critical { border-left-color: #FF3366; background: linear-gradient(90deg, rgba(255,51,102,0.06), rgba(21,28,40,0.65)); }
.alert-warning { border-left-color: #FFB800; background: linear-gradient(90deg, rgba(255,184,0,0.06), rgba(21,28,40,0.65)); }
.alert-info { border-left-color: #00D4FF; background: linear-gradient(90deg, rgba(0,212,255,0.06), rgba(21,28,40,0.65)); }
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 0.85rem;
}
th {
    background: #111820;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    color: #7a8899;
    border-bottom: 2px solid #1e2a3a;
    text-transform: uppercase;
    font-size: 0.7rem;
    letter-spacing: 0.06em;
}
td {
    padding: 8px 14px;
    border-bottom: 1px solid #1e2a3a;
}
tr:hover { background: rgba(0,212,255,0.03); }
.bar-container {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 4px 0;
}
.bar-label {
    min-width: 200px;
    font-size: 0.8rem;
    text-align: right;
    color: #7a8899;
}
.bar-fill {
    height: 20px;
    border-radius: 3px;
    min-width: 2px;
}
.bar-value {
    font-size: 0.75rem;
    font-family: monospace;
    color: #E6EDF3;
    min-width: 60px;
}
.footer {
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid #1e2a3a;
    font-size: 0.7rem;
    color: #4a5568;
    text-align: center;
}
.divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e2a3a, transparent);
    margin: 24px 0;
}
@media print {
    body { background: #0a0e14 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
}
"""


def html_wrapper(title: str, body_html: str, css: str = BASE_CSS) -> str:
    """Wrap content in a full HTML document."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<div class="container">
<div class="header-bar">
    <div>
        <div class="header-title">ORBITAL SENTINEL</div>
        <h1 style="margin:4px 0 0;border:none;padding:0;">{title}</h1>
    </div>
    <div class="header-time">Generated: {now}</div>
</div>
{body_html}
<div class="footer">
    Orbital Sentinel v1.0.0 &middot; Generated {now} &middot;
    This report is for informational purposes. Always verify with operational systems before taking action.
</div>
</div>
</body>
</html>"""


def risk_badge_html(category: str) -> str:
    """Return a colored risk badge HTML element."""
    cat = category.upper() if category else "NEGLIGIBLE"
    css_map = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low", "NEGLIGIBLE": "risk-negligible"}
    css_class = css_map.get(cat, "risk-negligible")
    return f'<span class="risk-badge {css_class}">{cat}</span>'


def metric_card_html(label: str, value: str, color: str = "") -> str:
    """Return a metric card HTML element."""
    color_class = f' class="metric-value {color}"' if color else ' class="metric-value"'
    return f"""<div class="metric-card">
    <div class="metric-label">{label}</div>
    <div{color_class}>{value}</div>
</div>"""


def table_html(headers: list[str], rows: list[list], highlight_col: int = None) -> str:
    """Return a styled HTML table."""
    header_cells = "".join(f"<th>{h}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = []
        for i, v in enumerate(row):
            style = ' style="color:#00D4FF;font-weight:600;"' if i == highlight_col else ""
            cells.append(f"<td{style}>{v}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{header_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def bar_chart_html(labels: list[str], values: list[float], colors: list[str] = None, max_width: int = 400) -> str:
    """Return a pure CSS horizontal bar chart."""
    if not values:
        return ""
    max_val = max(abs(v) for v in values) or 1
    if colors is None:
        colors = ["#00D4FF"] * len(values)
    bars = []
    for label, val, color in zip(labels, values, colors):
        width = int(abs(val) / max_val * max_width)
        bars.append(f"""<div class="bar-container">
    <div class="bar-label">{label}</div>
    <div class="bar-fill" style="width:{width}px;background:{color};"></div>
    <div class="bar-value">{val:.4f}</div>
</div>""")
    return "\n".join(bars)


def alert_html(title: str, detail: str, level: str = "info") -> str:
    """Return an alert card HTML element."""
    return f"""<div class="alert-card alert-{level}">
    <strong>{title}</strong><br>
    <span style="color:#8B949E;font-size:0.85rem;">{detail}</span>
</div>"""

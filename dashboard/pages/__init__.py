"""Dashboard page modules for Orbital Sentinel."""

from dashboard.pages.ensemble_comparison import render_ensemble_comparison
from dashboard.pages.alert_dashboard import render_alert_dashboard
from dashboard.pages.report_generator import render_report_generator
from dashboard.pages.system_status import render_system_status

__all__ = [
    "render_ensemble_comparison",
    "render_alert_dashboard",
    "render_report_generator",
    "render_system_status",
]

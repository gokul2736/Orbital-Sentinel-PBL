"""Alert management — evaluate rules and produce structured alerts."""

from datetime import datetime, timezone
from typing import Optional

from orbital_sentinel.alerts.rules import AlertRule, DEFAULT_RULES


class AlertManager:

    def __init__(self, rules: Optional[list[AlertRule]] = None):
        self.rules = rules or list(DEFAULT_RULES)

    def check(self, prediction_result: dict) -> list[dict]:
        """Evaluate all rules against a prediction result."""
        alerts = []
        for rule in self.rules:
            triggered, detail = rule.check_fn(prediction_result)
            if triggered:
                alerts.append({
                    "rule": rule.name,
                    "severity": rule.severity,
                    "title": rule.description,
                    "detail": detail,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_id": prediction_result.get("event_id"),
                })
        return sorted(alerts, key=lambda a: {"CRITICAL": 0, "WARNING": 1, "INFO": 2}.get(a["severity"], 3))

    def check_event_sequence(self, event_id: str, cdm_sequence: list[dict]) -> list[dict]:
        """Check event-level rules on a CDM sequence."""
        alerts = []
        if len(cdm_sequence) < 2:
            return alerts

        risks = [c.get("risk", -30) for c in cdm_sequence]
        latest = risks[-1]
        previous = risks[-2]

        if latest > previous and latest > -7:
            alerts.append({
                "rule": "escalation",
                "severity": "CRITICAL",
                "title": "Risk escalation detected",
                "detail": f"Event {event_id}: risk increased from {previous:.2f} to {latest:.2f}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_id": event_id,
            })
        elif latest > previous:
            alerts.append({
                "rule": "escalation_minor",
                "severity": "WARNING",
                "title": "Risk trending upward",
                "detail": f"Event {event_id}: risk increased from {previous:.2f} to {latest:.2f}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_id": event_id,
            })

        return alerts

    def format_alert(self, alert: dict) -> str:
        """Format a single alert as a human-readable string."""
        severity = alert.get("severity", "INFO")
        title = alert.get("title", "")
        detail = alert.get("detail", "")
        return f"[{severity}] {title}: {detail}"

    def format_alerts_html(self, alerts: list[dict]) -> str:
        """Format alerts as HTML."""
        from orbital_sentinel.reporting.templates import alert_html
        level_map = {"CRITICAL": "critical", "WARNING": "warning", "INFO": "info"}
        parts = []
        for a in alerts:
            level = level_map.get(a.get("severity", "INFO"), "info")
            parts.append(alert_html(a.get("title", ""), a.get("detail", ""), level))
        return "\n".join(parts)

"""Notification dispatch — log, webhook (Slack/Discord/Teams)."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class Notifier:

    def __init__(self, webhook_url: Optional[str] = None, log_file: str = "alerts.log"):
        self.webhook_url = webhook_url
        self.log_file = Path(log_file)

    def notify(self, alerts: list[dict]) -> None:
        """Dispatch alerts to all configured channels."""
        for alert in alerts:
            self._log_alert(alert)
            if self.webhook_url:
                self._send_webhook(alert)

    def _log_alert(self, alert: dict) -> None:
        """Write alert to log file."""
        timestamp = alert.get("timestamp", datetime.now(timezone.utc).isoformat())
        severity = alert.get("severity", "INFO")
        title = alert.get("title", "")
        detail = alert.get("detail", "")
        line = f"[{timestamp}] [{severity}] {title}: {detail}\n"

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            logger.error("Failed to write alert log: %s", e)

    def _send_webhook(self, alert: dict) -> None:
        """POST alert to a webhook URL (Slack/Discord/Teams compatible)."""
        payload = self._format_slack_payload(alert)
        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )
            if resp.status_code >= 400:
                logger.warning("Webhook returned %d: %s", resp.status_code, resp.text[:200])
        except requests.RequestException as e:
            logger.warning("Webhook delivery failed: %s", e)

    def _format_slack_payload(self, alert: dict) -> dict:
        """Format alert as a Slack-compatible webhook payload."""
        severity = alert.get("severity", "INFO")
        color_map = {"CRITICAL": "#FF3366", "WARNING": "#FFB800", "INFO": "#00D4FF"}
        color = color_map.get(severity, "#7a8899")
        emoji_map = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}
        emoji = emoji_map.get(severity, "ℹ️")

        event_id = alert.get("event_id")
        event_text = f" (Event: {event_id})" if event_id else ""

        return {
            "text": f"{emoji} *[{severity}]* {alert.get('title', '')}{event_text}",
            "attachments": [{
                "color": color,
                "text": alert.get("detail", ""),
                "footer": "Orbital Sentinel Alert System",
                "ts": int(datetime.now(timezone.utc).timestamp()),
            }],
        }

"""Alert system package."""

from orbital_sentinel.alerts.rules import AlertRule, DEFAULT_RULES
from orbital_sentinel.alerts.manager import AlertManager

__all__ = ["AlertRule", "AlertManager", "DEFAULT_RULES"]

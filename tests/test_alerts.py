"""Tests for the alert system."""

import pytest
import math


def test_high_risk_rule_triggers():
    from orbital_sentinel.alerts.rules import HIGH_RISK_RULE

    triggered, detail = HIGH_RISK_RULE.check_fn({"prediction": -3.0})
    assert triggered is True
    assert "-3.0" in detail


def test_high_risk_rule_no_trigger():
    from orbital_sentinel.alerts.rules import HIGH_RISK_RULE

    triggered, _ = HIGH_RISK_RULE.check_fn({"prediction": -10.0})
    assert triggered is False


def test_medium_risk_rule():
    from orbital_sentinel.alerts.rules import MEDIUM_RISK_RULE

    triggered, _ = MEDIUM_RISK_RULE.check_fn({"prediction": -6.0})
    assert triggered is True

    triggered, _ = MEDIUM_RISK_RULE.check_fn({"prediction": -3.0})
    assert triggered is False


def test_low_confidence_rule():
    from orbital_sentinel.alerts.rules import LOW_CONFIDENCE_RULE

    triggered, _ = LOW_CONFIDENCE_RULE.check_fn({"confidence": 0.1})
    assert triggered is True

    triggered, _ = LOW_CONFIDENCE_RULE.check_fn({"confidence": 0.8})
    assert triggered is False


def test_physics_disagreement_rule():
    from orbital_sentinel.alerts.rules import PHYSICS_DISAGREEMENT_RULE

    result = {"prediction": -5.0, "physics_result": {"log10_pc": -12.0}}
    triggered, detail = PHYSICS_DISAGREEMENT_RULE.check_fn(result)
    assert triggered is True
    assert "7.0" in detail


def test_physics_disagreement_no_physics():
    from orbital_sentinel.alerts.rules import PHYSICS_DISAGREEMENT_RULE

    triggered, _ = PHYSICS_DISAGREEMENT_RULE.check_fn({"prediction": -5.0})
    assert triggered is False


def test_covariance_invalid_rule():
    from orbital_sentinel.alerts.rules import COVARIANCE_INVALID_RULE

    result = {"physics_result": {"covariance_valid": False, "covariance_warnings": ["bad eigenvalue"]}}
    triggered, detail = COVARIANCE_INVALID_RULE.check_fn(result)
    assert triggered is True
    assert "eigenvalue" in detail


def test_wide_interval_rule():
    from orbital_sentinel.alerts.rules import WIDE_INTERVAL_RULE

    triggered, _ = WIDE_INTERVAL_RULE.check_fn({"interval": (-15.0, -5.0)})
    assert triggered is True

    triggered, _ = WIDE_INTERVAL_RULE.check_fn({"interval": (-8.0, -6.0)})
    assert triggered is False


def test_alert_manager_check():
    from orbital_sentinel.alerts.manager import AlertManager

    manager = AlertManager()
    alerts = manager.check({"prediction": -3.0, "confidence": 0.1})

    assert len(alerts) >= 2
    severities = [a["severity"] for a in alerts]
    assert "CRITICAL" in severities


def test_alert_manager_event_sequence():
    from orbital_sentinel.alerts.manager import AlertManager

    manager = AlertManager()
    sequence = [
        {"risk": -10.0},
        {"risk": -6.0},
    ]
    alerts = manager.check_event_sequence("EVT-1", sequence)
    assert len(alerts) >= 1
    assert alerts[0]["severity"] == "CRITICAL"


def test_alert_manager_format():
    from orbital_sentinel.alerts.manager import AlertManager

    manager = AlertManager()
    alert = {"severity": "CRITICAL", "title": "Test", "detail": "Details here"}
    formatted = manager.format_alert(alert)
    assert "[CRITICAL]" in formatted
    assert "Test" in formatted


def test_notifier_log(tmp_path):
    from orbital_sentinel.alerts.notifier import Notifier

    log_file = tmp_path / "test_alerts.log"
    notifier = Notifier(log_file=str(log_file))

    alerts = [{"severity": "CRITICAL", "title": "Test Alert", "detail": "Something happened", "timestamp": "2024-01-01T00:00:00"}]
    notifier.notify(alerts)

    content = log_file.read_text()
    assert "CRITICAL" in content
    assert "Test Alert" in content


def test_notifier_slack_payload():
    from orbital_sentinel.alerts.notifier import Notifier

    notifier = Notifier()
    alert = {"severity": "WARNING", "title": "Test", "detail": "Detail", "event_id": "EVT-1"}
    payload = notifier._format_slack_payload(alert)

    assert "text" in payload
    assert "attachments" in payload
    assert "EVT-1" in payload["text"]

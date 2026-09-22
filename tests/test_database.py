"""Tests for the Orbital Sentinel database persistence layer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
from datetime import datetime, timezone

import pytest

from orbital_sentinel.database import (
    Database,
    AlertRecord,
    CDMRecord,
    EventRecord,
    ModelRecord,
    PredictionRecord,
)
from orbital_sentinel.database.repository import (
    AlertRepository,
    CDMRepository,
    EventRepository,
    ModelRepository,
    PredictionRepository,
)
from orbital_sentinel.database.migrations import run_migrations


@pytest.fixture
def db():
    database = Database(url="sqlite:///:memory:")
    database.create_tables()
    yield database
    database.close()


class TestDatabase:
    def test_create_tables(self, db: Database):
        from sqlalchemy import inspect

        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        assert "cdm_records" in tables
        assert "prediction_records" in tables
        assert "event_records" in tables
        assert "model_records" in tables
        assert "alert_records" in tables
        assert "schema_version" in tables

    def test_session_context_manager(self, db: Database):
        with db.get_session() as session:
            assert session is not None

    def test_session_rollback_on_error(self, db: Database):
        try:
            with db.get_session() as session:
                record = CDMRecord(source="test", cdm_id="dup-1")
                session.add(record)
                session.flush()
                duplicate = CDMRecord(source="test", cdm_id="dup-1")
                session.add(duplicate)
                session.flush()
        except Exception:
            pass

        with db.get_session() as session:
            count = CDMRepository.count_cdms(session)
            assert count <= 1


class TestCDMRepository:
    def test_save_and_get_cdm(self, db: Database):
        with db.get_session() as session:
            cdm = CDMRepository.save_cdm(session, {
                "cdm_id": "CDM-001",
                "event_id": "EVT-001",
                "miss_distance": 150.5,
                "relative_speed": 12.3,
                "risk": -3.5,
                "sat1_name": "SAT-A",
                "sat2_name": "SAT-B",
                "raw_data": {"key": "value"},
            }, source="space_track")
            assert cdm.id is not None
            assert cdm.cdm_id == "CDM-001"

        with db.get_session() as session:
            fetched = CDMRepository.get_cdm(session, cdm.id)
            assert fetched is not None
            assert fetched.cdm_id == "CDM-001"
            assert fetched.miss_distance == 150.5
            assert fetched.source == "space_track"

    def test_get_cdms_by_event(self, db: Database):
        with db.get_session() as session:
            CDMRepository.save_cdm(session, {"cdm_id": "C1", "event_id": "E1"}, source="test")
            CDMRepository.save_cdm(session, {"cdm_id": "C2", "event_id": "E1"}, source="test")
            CDMRepository.save_cdm(session, {"cdm_id": "C3", "event_id": "E2"}, source="test")

        with db.get_session() as session:
            results = CDMRepository.get_cdms_by_event(session, "E1")
            assert len(results) == 2

    def test_get_recent_cdms(self, db: Database):
        with db.get_session() as session:
            for i in range(5):
                CDMRepository.save_cdm(session, {"cdm_id": f"R-{i}"}, source="test")

        with db.get_session() as session:
            results = CDMRepository.get_recent_cdms(session, limit=3)
            assert len(results) == 3

    def test_count_cdms(self, db: Database):
        with db.get_session() as session:
            assert CDMRepository.count_cdms(session) == 0
            CDMRepository.save_cdm(session, {"cdm_id": "X1"}, source="test")
            CDMRepository.save_cdm(session, {"cdm_id": "X2"}, source="test")

        with db.get_session() as session:
            assert CDMRepository.count_cdms(session) == 2

    def test_raw_and_mapped_data_json(self, db: Database):
        with db.get_session() as session:
            cdm = CDMRepository.save_cdm(session, {
                "cdm_id": "JSON-1",
                "raw_data": {"field1": "val1", "nested": [1, 2]},
                "mapped_data": {"esa_field": "mapped_val"},
            }, source="test")

        with db.get_session() as session:
            fetched = CDMRepository.get_cdm(session, cdm.id)
            raw = fetched.get_raw_data()
            mapped = fetched.get_mapped_data()
            assert raw["field1"] == "val1"
            assert raw["nested"] == [1, 2]
            assert mapped["esa_field"] == "mapped_val"


class TestPredictionRepository:
    def _create_cdm(self, db: Database) -> int:
        with db.get_session() as session:
            cdm = CDMRepository.save_cdm(session, {"cdm_id": "PRED-CDM"}, source="test")
            return cdm.id

    def test_save_and_get_prediction(self, db: Database):
        cdm_id = self._create_cdm(db)
        with db.get_session() as session:
            pred = PredictionRepository.save_prediction(session, cdm_id, {
                "model_name": "xgboost_v1",
                "predicted_risk": -3.2,
                "risk_category": "HIGH",
                "confidence": 0.92,
                "interval_lower": -4.0,
                "interval_upper": -2.5,
                "inference_time_ms": 12.5,
                "top_features": {"miss_distance": 0.45, "relative_speed": 0.30},
            })
            assert pred.id is not None

        with db.get_session() as session:
            preds = PredictionRepository.get_predictions_for_cdm(session, cdm_id)
            assert len(preds) == 1
            assert preds[0].model_name == "xgboost_v1"
            assert preds[0].confidence == 0.92

    def test_cdm_prediction_relationship(self, db: Database):
        cdm_id = self._create_cdm(db)
        with db.get_session() as session:
            PredictionRepository.save_prediction(session, cdm_id, {
                "model_name": "model_a",
                "predicted_risk": -3.0,
                "risk_category": "HIGH",
            })
            PredictionRepository.save_prediction(session, cdm_id, {
                "model_name": "model_b",
                "predicted_risk": -5.0,
                "risk_category": "MEDIUM",
            })

        with db.get_session() as session:
            cdm = CDMRepository.get_cdm(session, cdm_id)
            assert len(cdm.predictions) == 2

    def test_high_risk_predictions(self, db: Database):
        cdm_id = self._create_cdm(db)
        with db.get_session() as session:
            PredictionRepository.save_prediction(session, cdm_id, {
                "model_name": "m1", "predicted_risk": -2.0, "risk_category": "HIGH",
            })
            PredictionRepository.save_prediction(session, cdm_id, {
                "model_name": "m2", "predicted_risk": -8.0, "risk_category": "NEGLIGIBLE",
            })

        with db.get_session() as session:
            high = PredictionRepository.get_high_risk_predictions(session, threshold=-5.0)
            assert len(high) == 1
            assert high[0].predicted_risk == -2.0

    def test_prediction_stats(self, db: Database):
        cdm_id = self._create_cdm(db)
        with db.get_session() as session:
            PredictionRepository.save_prediction(session, cdm_id, {
                "model_name": "m", "predicted_risk": -3.0, "risk_category": "HIGH", "confidence": 0.9,
            })
            PredictionRepository.save_prediction(session, cdm_id, {
                "model_name": "m", "predicted_risk": -6.0, "risk_category": "LOW", "confidence": 0.8,
            })

        with db.get_session() as session:
            stats = PredictionRepository.get_prediction_stats(session)
            assert stats["total"] == 2
            assert stats["by_category"]["HIGH"] == 1
            assert stats["by_category"]["LOW"] == 1
            assert stats["avg_confidence"] == 0.85

    def test_prediction_top_features_json(self, db: Database):
        cdm_id = self._create_cdm(db)
        features = {"miss_distance": 0.45, "speed": 0.30}
        with db.get_session() as session:
            pred = PredictionRepository.save_prediction(session, cdm_id, {
                "model_name": "m", "predicted_risk": -3.0, "risk_category": "HIGH",
                "top_features": features,
            })
            pred_id = pred.id

        with db.get_session() as session:
            preds = PredictionRepository.get_predictions_for_cdm(session, cdm_id)
            assert preds[0].get_top_features() == features


class TestEventRepository:
    def test_create_event(self, db: Database):
        with db.get_session() as session:
            event = EventRepository.upsert_event(session, "EVT-100", risk=-3.0, cdm_count=1)
            assert event.event_id == "EVT-100"
            assert event.latest_risk == -3.0
            assert event.peak_risk == -3.0
            assert event.status == "ACTIVE"

    def test_upsert_updates_existing(self, db: Database):
        with db.get_session() as session:
            EventRepository.upsert_event(session, "EVT-200", risk=-5.0, cdm_count=1)

        with db.get_session() as session:
            event = EventRepository.upsert_event(session, "EVT-200", risk=-3.0, cdm_count=3)
            assert event.cdm_count == 3
            assert event.latest_risk == -3.0
            assert event.peak_risk == -3.0
            assert event.risk_trend == "INCREASING"

    def test_upsert_tracks_peak_risk(self, db: Database):
        with db.get_session() as session:
            EventRepository.upsert_event(session, "EVT-300", risk=-2.0, cdm_count=1)

        with db.get_session() as session:
            event = EventRepository.upsert_event(session, "EVT-300", risk=-5.0, cdm_count=2)
            assert event.latest_risk == -5.0
            assert event.peak_risk == -2.0
            assert event.risk_trend == "DECREASING"

    def test_get_active_events(self, db: Database):
        with db.get_session() as session:
            EventRepository.upsert_event(session, "A1", risk=-3.0, cdm_count=1)
            EventRepository.upsert_event(session, "A2", risk=-4.0, cdm_count=1)

        with db.get_session() as session:
            active = EventRepository.get_active_events(session)
            assert len(active) == 2

    def test_get_escalating_events(self, db: Database):
        with db.get_session() as session:
            EventRepository.upsert_event(session, "ESC-1", risk=-5.0, cdm_count=1)

        with db.get_session() as session:
            EventRepository.upsert_event(session, "ESC-1", risk=-3.0, cdm_count=2)

        with db.get_session() as session:
            escalating = EventRepository.get_escalating_events(session)
            assert len(escalating) == 1
            assert escalating[0].event_id == "ESC-1"


class TestAlertRepository:
    def test_create_alert(self, db: Database):
        with db.get_session() as session:
            alert = AlertRepository.create_alert(
                session,
                alert_type="HIGH_RISK",
                severity="CRITICAL",
                title="High collision risk detected",
                detail="Event EVT-001 risk is -2.1",
                event_id="EVT-001",
            )
            assert alert.id is not None
            assert alert.acknowledged is False

    def test_acknowledge_alert(self, db: Database):
        with db.get_session() as session:
            alert = AlertRepository.create_alert(
                session, alert_type="NEW_EVENT", severity="INFO",
                title="New event", detail="Details",
            )
            alert_id = alert.id

        with db.get_session() as session:
            acked = AlertRepository.acknowledge_alert(session, alert_id)
            assert acked.acknowledged is True
            assert acked.acknowledged_at is not None

    def test_get_unacknowledged(self, db: Database):
        with db.get_session() as session:
            AlertRepository.create_alert(session, "A", "INFO", "T1", "D1")
            a2 = AlertRepository.create_alert(session, "B", "WARNING", "T2", "D2")
            a2.acknowledged = True

        with db.get_session() as session:
            unacked = AlertRepository.get_unacknowledged(session)
            assert len(unacked) == 1
            assert unacked[0].title == "T1"

    def test_get_recent_alerts(self, db: Database):
        with db.get_session() as session:
            for i in range(5):
                AlertRepository.create_alert(session, "TYPE", "INFO", f"Alert {i}", "detail")

        with db.get_session() as session:
            recent = AlertRepository.get_recent_alerts(session, limit=3)
            assert len(recent) == 3

    def test_acknowledge_nonexistent_raises(self, db: Database):
        with db.get_session() as session:
            with pytest.raises(ValueError, match="not found"):
                AlertRepository.acknowledge_alert(session, 9999)


class TestModelRepository:
    def test_register_model(self, db: Database):
        with db.get_session() as session:
            model = ModelRepository.register_model(session, {
                "model_name": "xgb_v1",
                "model_type": "XGBoost",
                "train_rmse": 0.15,
                "val_rmse": 0.18,
                "train_r2": 0.95,
                "val_r2": 0.92,
                "feature_count": 42,
                "artifact_path": "models/xgb_v1.pkl",
                "metadata": {"notes": "first model"},
            })
            assert model.id is not None
            assert model.model_name == "xgb_v1"
            assert model.is_active is False

    def test_set_active_model(self, db: Database):
        with db.get_session() as session:
            m1 = ModelRepository.register_model(session, {
                "model_name": "m1", "model_type": "XGBoost", "is_active": True,
            })
            m2 = ModelRepository.register_model(session, {
                "model_name": "m2", "model_type": "LightGBM",
            })

        with db.get_session() as session:
            ModelRepository.set_active_model(session, m2.id)

        with db.get_session() as session:
            active = ModelRepository.get_active_model(session)
            assert active is not None
            assert active.model_name == "m2"

            all_models = ModelRepository.get_all_models(session)
            active_count = sum(1 for m in all_models if m.is_active)
            assert active_count == 1

    def test_get_all_models(self, db: Database):
        with db.get_session() as session:
            ModelRepository.register_model(session, {"model_name": "a", "model_type": "XGBoost"})
            ModelRepository.register_model(session, {"model_name": "b", "model_type": "LightGBM"})

        with db.get_session() as session:
            models = ModelRepository.get_all_models(session)
            assert len(models) == 2

    def test_set_active_nonexistent_raises(self, db: Database):
        with db.get_session() as session:
            with pytest.raises(ValueError, match="not found"):
                ModelRepository.set_active_model(session, 9999)

    def test_model_metadata_json(self, db: Database):
        meta = {"version": "1.0", "tags": ["production"]}
        with db.get_session() as session:
            model = ModelRepository.register_model(session, {
                "model_name": "meta_test", "model_type": "XGBoost", "metadata": meta,
            })
            model_id = model.id

        with db.get_session() as session:
            m = session.get(ModelRecord, model_id)
            assert m.get_metadata() == meta


class TestMigrations:
    def test_run_migrations_on_empty_db(self):
        db = Database(url="sqlite:///:memory:")
        version = run_migrations(db)
        assert version >= 1

        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        assert "cdm_records" in tables
        assert "schema_version" in tables
        db.close()

    def test_run_migrations_idempotent(self):
        db = Database(url="sqlite:///:memory:")
        v1 = run_migrations(db)
        v2 = run_migrations(db)
        assert v1 == v2
        db.close()

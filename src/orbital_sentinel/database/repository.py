"""Repository pattern CRUD operations for Orbital Sentinel."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from .models import AlertRecord, CDMRecord, EventRecord, ModelRecord, PredictionRecord


class CDMRepository:
    """CRUD operations for CDM records."""

    @staticmethod
    def save_cdm(session: Session, cdm_data: dict, source: str) -> CDMRecord:
        record = CDMRecord(
            cdm_id=cdm_data.get("cdm_id"),
            event_id=cdm_data.get("event_id"),
            source=source,
            tca=cdm_data.get("tca"),
            miss_distance=cdm_data.get("miss_distance"),
            relative_speed=cdm_data.get("relative_speed"),
            risk=cdm_data.get("risk"),
            sat1_name=cdm_data.get("sat1_name"),
            sat2_name=cdm_data.get("sat2_name"),
            raw_data=json.dumps(cdm_data.get("raw_data")) if cdm_data.get("raw_data") else None,
            mapped_data=json.dumps(cdm_data.get("mapped_data")) if cdm_data.get("mapped_data") else None,
        )
        session.add(record)
        session.flush()
        return record

    @staticmethod
    def get_cdm(session: Session, cdm_id: int) -> Optional[CDMRecord]:
        return session.get(CDMRecord, cdm_id)

    @staticmethod
    def get_cdms_by_event(session: Session, event_id: str) -> List[CDMRecord]:
        stmt = select(CDMRecord).where(CDMRecord.event_id == event_id).order_by(CDMRecord.created_at.desc())
        return list(session.scalars(stmt).all())

    @staticmethod
    def get_recent_cdms(session: Session, limit: int = 50) -> List[CDMRecord]:
        stmt = select(CDMRecord).order_by(CDMRecord.created_at.desc()).limit(limit)
        return list(session.scalars(stmt).all())

    @staticmethod
    def count_cdms(session: Session) -> int:
        stmt = select(func.count(CDMRecord.id))
        return session.scalar(stmt) or 0


class PredictionRepository:
    """CRUD operations for prediction records."""

    @staticmethod
    def save_prediction(session: Session, cdm_id: int, prediction_data: dict) -> PredictionRecord:
        record = PredictionRecord(
            cdm_record_id=cdm_id,
            model_name=prediction_data["model_name"],
            predicted_risk=prediction_data["predicted_risk"],
            risk_category=prediction_data["risk_category"],
            confidence=prediction_data.get("confidence"),
            interval_lower=prediction_data.get("interval_lower"),
            interval_upper=prediction_data.get("interval_upper"),
            physics_log10_pc=prediction_data.get("physics_log10_pc"),
            fused_risk=prediction_data.get("fused_risk"),
            explanation=prediction_data.get("explanation"),
            top_features=json.dumps(prediction_data["top_features"]) if prediction_data.get("top_features") else None,
            inference_time_ms=prediction_data.get("inference_time_ms"),
        )
        session.add(record)
        session.flush()
        return record

    @staticmethod
    def get_predictions_for_cdm(session: Session, cdm_id: int) -> List[PredictionRecord]:
        stmt = (
            select(PredictionRecord)
            .where(PredictionRecord.cdm_record_id == cdm_id)
            .order_by(PredictionRecord.created_at.desc())
        )
        return list(session.scalars(stmt).all())

    @staticmethod
    def get_recent_predictions(session: Session, limit: int = 50) -> List[PredictionRecord]:
        stmt = select(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(limit)
        return list(session.scalars(stmt).all())

    @staticmethod
    def get_high_risk_predictions(session: Session, threshold: float = -5.0) -> List[PredictionRecord]:
        stmt = (
            select(PredictionRecord)
            .where(PredictionRecord.predicted_risk >= threshold)
            .order_by(PredictionRecord.predicted_risk.desc())
        )
        return list(session.scalars(stmt).all())

    @staticmethod
    def get_prediction_stats(session: Session) -> dict:
        total = session.scalar(select(func.count(PredictionRecord.id))) or 0
        if total == 0:
            return {"total": 0, "by_category": {}, "avg_confidence": None}

        avg_conf = session.scalar(select(func.avg(PredictionRecord.confidence)))

        category_counts: Dict[str, int] = {}
        stmt = (
            select(PredictionRecord.risk_category, func.count(PredictionRecord.id))
            .group_by(PredictionRecord.risk_category)
        )
        for category, count in session.execute(stmt).all():
            category_counts[category] = count

        return {
            "total": total,
            "by_category": category_counts,
            "avg_confidence": round(avg_conf, 4) if avg_conf is not None else None,
        }


class EventRepository:
    """CRUD operations for conjunction event records."""

    @staticmethod
    def upsert_event(session: Session, event_id: str, risk: float, cdm_count: int) -> EventRecord:
        existing = session.scalar(select(EventRecord).where(EventRecord.event_id == event_id))
        now = datetime.now(timezone.utc)

        if existing is not None:
            previous_risk = existing.latest_risk
            existing.latest_risk = risk
            existing.cdm_count = cdm_count
            existing.last_updated = now
            if existing.peak_risk is None or risk > existing.peak_risk:
                existing.peak_risk = risk
            if previous_risk is not None:
                if risk > previous_risk:
                    existing.risk_trend = "INCREASING"
                elif risk < previous_risk:
                    existing.risk_trend = "DECREASING"
                else:
                    existing.risk_trend = "STABLE"
            session.flush()
            return existing

        record = EventRecord(
            event_id=event_id,
            first_seen=now,
            last_updated=now,
            cdm_count=cdm_count,
            latest_risk=risk,
            peak_risk=risk,
            status="ACTIVE",
        )
        session.add(record)
        session.flush()
        return record

    @staticmethod
    def get_event(session: Session, event_id: str) -> Optional[EventRecord]:
        return session.scalar(select(EventRecord).where(EventRecord.event_id == event_id))

    @staticmethod
    def get_active_events(session: Session) -> List[EventRecord]:
        stmt = (
            select(EventRecord)
            .where(EventRecord.status == "ACTIVE")
            .order_by(EventRecord.last_updated.desc())
        )
        return list(session.scalars(stmt).all())

    @staticmethod
    def get_escalating_events(session: Session) -> List[EventRecord]:
        stmt = (
            select(EventRecord)
            .where(EventRecord.status == "ACTIVE", EventRecord.risk_trend == "INCREASING")
            .order_by(EventRecord.latest_risk.desc())
        )
        return list(session.scalars(stmt).all())


class AlertRepository:
    """CRUD operations for alert records."""

    @staticmethod
    def create_alert(
        session: Session,
        alert_type: str,
        severity: str,
        title: str,
        detail: str,
        event_id: Optional[str] = None,
    ) -> AlertRecord:
        record = AlertRecord(
            event_id=event_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            detail=detail,
        )
        session.add(record)
        session.flush()
        return record

    @staticmethod
    def get_unacknowledged(session: Session) -> List[AlertRecord]:
        stmt = (
            select(AlertRecord)
            .where(AlertRecord.acknowledged == False)  # noqa: E712
            .order_by(AlertRecord.created_at.desc())
        )
        return list(session.scalars(stmt).all())

    @staticmethod
    def acknowledge_alert(session: Session, alert_id: int) -> AlertRecord:
        record = session.get(AlertRecord, alert_id)
        if record is None:
            raise ValueError(f"Alert {alert_id} not found")
        record.acknowledged = True
        record.acknowledged_at = datetime.now(timezone.utc)
        session.flush()
        return record

    @staticmethod
    def get_recent_alerts(session: Session, limit: int = 20) -> List[AlertRecord]:
        stmt = select(AlertRecord).order_by(AlertRecord.created_at.desc()).limit(limit)
        return list(session.scalars(stmt).all())


class ModelRepository:
    """CRUD operations for model registry records."""

    @staticmethod
    def register_model(session: Session, model_data: dict) -> ModelRecord:
        record = ModelRecord(
            model_name=model_data["model_name"],
            model_type=model_data["model_type"],
            train_rmse=model_data.get("train_rmse"),
            val_rmse=model_data.get("val_rmse"),
            train_r2=model_data.get("train_r2"),
            val_r2=model_data.get("val_r2"),
            feature_count=model_data.get("feature_count"),
            artifact_path=model_data.get("artifact_path"),
            is_active=model_data.get("is_active", False),
            metadata_json=json.dumps(model_data["metadata"]) if model_data.get("metadata") else None,
        )
        session.add(record)
        session.flush()
        return record

    @staticmethod
    def get_active_model(session: Session) -> Optional[ModelRecord]:
        return session.scalar(select(ModelRecord).where(ModelRecord.is_active == True))  # noqa: E712

    @staticmethod
    def set_active_model(session: Session, model_id: int) -> ModelRecord:
        session.execute(update(ModelRecord).values(is_active=False))
        record = session.get(ModelRecord, model_id)
        if record is None:
            raise ValueError(f"Model {model_id} not found")
        record.is_active = True
        session.flush()
        return record

    @staticmethod
    def get_all_models(session: Session) -> List[ModelRecord]:
        stmt = select(ModelRecord).order_by(ModelRecord.trained_at.desc())
        return list(session.scalars(stmt).all())

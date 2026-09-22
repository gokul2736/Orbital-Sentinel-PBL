"""SQLAlchemy ORM models for Orbital Sentinel."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class CDMRecord(Base):
    __tablename__ = "cdm_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cdm_id: Mapped[Optional[str]] = mapped_column(String(256), unique=True, nullable=True)
    event_id: Mapped[Optional[str]] = mapped_column(String(256), index=True, nullable=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    tca: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    miss_distance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    relative_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sat1_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    sat2_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    raw_data: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    mapped_data: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    predictions: Mapped[List[PredictionRecord]] = relationship(
        "PredictionRecord", back_populates="cdm_record", cascade="all, delete-orphan"
    )

    def set_raw_data(self, data: dict) -> None:
        self.raw_data = json.dumps(data)

    def get_raw_data(self) -> Optional[dict]:
        return json.loads(self.raw_data) if self.raw_data else None

    def set_mapped_data(self, data: dict) -> None:
        self.mapped_data = json.dumps(data)

    def get_mapped_data(self) -> Optional[dict]:
        return json.loads(self.mapped_data) if self.mapped_data else None


class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cdm_record_id: Mapped[int] = mapped_column(Integer, ForeignKey("cdm_records.id"), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    predicted_risk: Mapped[float] = mapped_column(Float, nullable=False)
    risk_category: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    interval_lower: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    interval_upper: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    physics_log10_pc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fused_risk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    top_features: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    inference_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    cdm_record: Mapped[CDMRecord] = relationship("CDMRecord", back_populates="predictions")

    def set_top_features(self, features: dict) -> None:
        self.top_features = json.dumps(features)

    def get_top_features(self) -> Optional[dict]:
        return json.loads(self.top_features) if self.top_features else None


class EventRecord(Base):
    __tablename__ = "event_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(256), unique=True, index=True, nullable=False)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    cdm_count: Mapped[int] = mapped_column(Integer, default=1)
    latest_risk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    peak_risk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_trend: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="ACTIVE")
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class ModelRecord(Base):
    __tablename__ = "model_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    train_rmse: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    val_rmse: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    train_r2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    val_r2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feature_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    artifact_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def set_metadata(self, data: dict) -> None:
        self.metadata_json = json.dumps(data)

    def get_metadata(self) -> Optional[dict]:
        return json.loads(self.metadata_json) if self.metadata_json else None


class AlertRecord(Base):
    __tablename__ = "alert_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[Optional[str]] = mapped_column(String(256), index=True, nullable=True)
    alert_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class SchemaVersion(Base):
    __tablename__ = "schema_version"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    description: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

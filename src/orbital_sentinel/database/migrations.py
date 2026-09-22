"""Simple schema migration utility for Orbital Sentinel."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import inspect, select

from .connection import Database
from .models import Base, SchemaVersion

CURRENT_VERSION = 1


def _get_current_version(db: Database) -> int:
    inspector = inspect(db.engine)
    if "schema_version" not in inspector.get_table_names():
        return 0
    with db.get_session() as session:
        stmt = select(SchemaVersion.version).order_by(SchemaVersion.id.desc()).limit(1)
        version = session.scalar(stmt)
        return version if version is not None else 0


def _record_version(db: Database, version: int, description: str) -> None:
    with db.get_session() as session:
        entry = SchemaVersion(
            version=version,
            applied_at=datetime.now(timezone.utc),
            description=description,
        )
        session.add(entry)


def _migration_v1(db: Database) -> None:
    """Initial schema: create all tables and indexes."""
    Base.metadata.create_all(db.engine)


_MIGRATIONS = {
    1: ("Initial schema creation", _migration_v1),
}


def run_migrations(db: Database) -> int:
    current = _get_current_version(db)

    if current == 0:
        Base.metadata.create_all(db.engine)

    for version in range(current + 1, CURRENT_VERSION + 1):
        description, migration_fn = _MIGRATIONS[version]
        migration_fn(db)
        _record_version(db, version, description)

    return CURRENT_VERSION

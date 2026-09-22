"""Database connection management for Orbital Sentinel."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


class Database:
    """Thread-safe database connection manager."""

    def __init__(self, url: str = "sqlite:///data/orbital_sentinel.db") -> None:
        connect_args = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        self._engine: Engine = create_engine(
            url,
            connect_args=connect_args,
            pool_pre_ping=True,
        )
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    @property
    def engine(self) -> Engine:
        return self._engine

    def create_tables(self) -> None:
        Base.metadata.create_all(self._engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self) -> None:
        self._engine.dispose()


_lock = threading.Lock()
_instance: Database | None = None


def get_db(url: str = "sqlite:///data/orbital_sentinel.db") -> Database:
    """Return a singleton Database instance (thread-safe)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = Database(url)
    return _instance


def reset_db() -> None:
    """Reset the singleton (useful for testing)."""
    global _instance
    with _lock:
        if _instance is not None:
            _instance.close()
        _instance = None

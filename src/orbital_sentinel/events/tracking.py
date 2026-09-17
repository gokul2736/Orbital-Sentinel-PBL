"""Track conjunction events across multiple CDM updates."""

from collections import defaultdict
from typing import Optional


class EventTracker:
    """Tracks conjunction events by ingesting CDM records and organizing them by event."""

    def __init__(self):
        self._events: dict[str, list[dict]] = defaultdict(list)

    def ingest_cdm(self, cdm_row: dict) -> str:
        """Add a CDM to the appropriate event. Returns event_id."""
        event_id = cdm_row.get("event_id")
        if event_id is None:
            raise ValueError("CDM row must contain 'event_id'")

        self._events[str(event_id)].append(dict(cdm_row))
        return str(event_id)

    def get_event_history(self, event_id) -> list:
        """Return list of CDMs for an event sorted by time_to_tca descending."""
        event_id = str(event_id)
        cdms = self._events.get(event_id, [])
        return sorted(cdms, key=lambda x: x.get("time_to_tca", 0), reverse=True)

    def get_active_events(self, max_tca_hours: float = 72) -> list:
        """Return events where latest CDM has time_to_tca < max_tca_hours.

        time_to_tca is assumed to be in days, so we convert max_tca_hours to days.
        """
        max_tca_days = max_tca_hours / 24.0
        active = []

        for event_id, cdms in self._events.items():
            if not cdms:
                continue
            latest = min(cdms, key=lambda x: x.get("time_to_tca", float("inf")))
            latest_tca = latest.get("time_to_tca")
            if latest_tca is not None and latest_tca < max_tca_days:
                active.append({
                    "event_id": event_id,
                    "n_cdms": len(cdms),
                    "latest_time_to_tca": latest_tca,
                    "latest_cdm": latest,
                })

        return sorted(active, key=lambda x: x["latest_time_to_tca"])

    @property
    def n_events(self) -> int:
        return len(self._events)

    @property
    def all_event_ids(self) -> list:
        return list(self._events.keys())

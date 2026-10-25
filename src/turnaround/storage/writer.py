"""SQLite event sink: persists domain events and maintains the sessions/intervals projections."""
from __future__ import annotations

import json
import logging
import sqlite3
from typing import Callable, Dict, Optional, Sequence

from turnaround.domain.events import DomainEvent, EventType
from turnaround.domain.types import RunInfo
from turnaround.storage.database import connect, initialize_schema

logger = logging.getLogger(__name__)


class SqliteEventWriter:
    """Implements the EventSink protocol on top of SQLite.

    Every event is appended to ``events``; session/interval events additionally update the
    ``sessions`` and ``service_intervals`` tables so readers never need to replay the log.
    """

    def __init__(self, db_path: str) -> None:
        """Open the database and create the schema.

        Args:
            db_path: SQLite file path.
        """
        self._db_path = db_path
        self._conn: sqlite3.Connection = connect(db_path)
        initialize_schema(self._conn)
        self._run_id: Optional[str] = None
        self._projections: Dict[EventType, Callable[[str, DomainEvent], None]] = {
            EventType.SESSION_STARTED: self._on_session_started,
            EventType.SESSION_ENDED: self._on_session_ended,
            EventType.SERVICE_STARTED: self._on_service_started,
            EventType.SERVICE_ENDED: self._on_service_ended,
            EventType.SERVICE_DISCARDED: self._on_service_discarded,
        }

    def begin_run(self, run: RunInfo) -> None:
        """Register the run.

        Args:
            run: Run metadata (config is stored as JSON).
        Side effects:
            Inserts a row into ``runs`` (replacing a row with the same id) and commits.
        """
        self._run_id = run.run_id
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, started_ts, ended_ts, source_uri, config_json) VALUES (?, ?, NULL, ?, ?)",
                (run.run_id, run.started_ts, run.source_uri, json.dumps(run.config, default=str)),
            )

    def handle(self, events: Sequence[DomainEvent]) -> None:
        """Persist a batch of events atomically.

        Args:
            events: Events in emission order.
        Raises:
            RuntimeError: If ``begin_run`` was not called.
            sqlite3.Error: On database failure (the batch is rolled back; the orchestrator logs and continues).
        Logic:
            In one transaction: insert each event into ``events`` then apply its projection (if any).
        """
        if not events:
            return
        if self._run_id is None:
            raise RuntimeError("begin_run() must be called before handle()")
        run_id = self._run_id
        with self._conn:
            for event in events:
                self._conn.execute(
                    "INSERT INTO events (run_id, ts, event_type, session_no, aircraft_id, service, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        event.ts,
                        event.event_type.value,
                        event.session_no,
                        event.aircraft_id,
                        event.service.value if event.service else None,
                        json.dumps(dict(event.payload), default=str),
                    ),
                )
                projection = self._projections.get(event.event_type)
                if projection is not None:
                    projection(run_id, event)

    def end_run(self, ended_ts: float) -> None:
        """Mark the run as ended.

        Args:
            ended_ts: Wall-clock end time.
        """
        if self._run_id is None:
            return
        with self._conn:
            self._conn.execute("UPDATE runs SET ended_ts = ? WHERE run_id = ?", (ended_ts, self._run_id))

    def close(self) -> None:
        """Close the database connection (idempotent)."""
        try:
            self._conn.close()
        except sqlite3.Error:  # pragma: no cover - defensive
            logger.debug("close failed", exc_info=True)

    def _on_session_started(self, run_id: str, event: DomainEvent) -> None:
        """Insert an open session row.

        Args:
            run_id: Run id.
            event: SESSION_STARTED event (payload: partial_start).
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions (run_id, session_no, aircraft_id, first_seen_ts, end_ts, status, closure_reason, partial_start, partial_end) "
            "VALUES (?, ?, ?, ?, NULL, 'open', NULL, ?, 0)",
            (run_id, event.session_no, event.aircraft_id, event.ts, int(bool(event.payload.get("partial_start", False)))),
        )

    def _on_session_ended(self, run_id: str, event: DomainEvent) -> None:
        """Close the session row.

        Args:
            run_id: Run id.
            event: SESSION_ENDED event (payload: reason, partial_end).
        """
        self._conn.execute(
            "UPDATE sessions SET end_ts = ?, status = 'closed', closure_reason = ?, partial_end = ? WHERE run_id = ? AND session_no = ?",
            (event.ts, event.payload.get("reason"), int(bool(event.payload.get("partial_end", False))), run_id, event.session_no),
        )

    def _on_service_started(self, run_id: str, event: DomainEvent) -> None:
        """Insert an open service interval.

        Args:
            run_id: Run id.
            event: SERVICE_STARTED event (ts = interval start).
        """
        self._conn.execute(
            "INSERT OR IGNORE INTO service_intervals (run_id, session_no, service, start_ts, end_ts, end_reason) VALUES (?, ?, ?, ?, NULL, NULL)",
            (run_id, event.session_no, event.service.value if event.service else None, event.ts),
        )

    def _on_service_ended(self, run_id: str, event: DomainEvent) -> None:
        """Close the open interval of that service.

        Args:
            run_id: Run id.
            event: SERVICE_ENDED event (ts = interval end, payload: end_reason).
        """
        self._conn.execute(
            "UPDATE service_intervals SET end_ts = ?, end_reason = ? WHERE run_id = ? AND session_no = ? AND service = ? AND end_ts IS NULL",
            (event.ts, event.payload.get("end_reason"), run_id, event.session_no, event.service.value if event.service else None),
        )

    def _on_service_discarded(self, run_id: str, event: DomainEvent) -> None:
        """Retract the open interval of that service (false trigger).

        Args:
            run_id: Run id.
            event: SERVICE_DISCARDED event.
        """
        self._conn.execute(
            "DELETE FROM service_intervals WHERE run_id = ? AND session_no = ? AND service = ? AND end_ts IS NULL",
            (run_id, event.session_no, event.service.value if event.service else None),
        )

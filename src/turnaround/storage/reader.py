"""Read-only SQLite access for the dashboard and offline analysis."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from turnaround.domain.records import EventRecord, IntervalRecord, RunRecord, SessionRecord
from turnaround.storage.database import connect


class AnalyticsReader:
    """Stateless query helper: every call opens, queries and closes its own read-only connection.

    This keeps the reader thread-safe (Streamlit reruns on different threads) and lets it run
    while the pipeline writes (WAL mode). A missing database yields empty results.
    """

    def __init__(self, db_path: str) -> None:
        """Remember the database location.

        Args:
            db_path: SQLite file path.
        """
        self._db_path = db_path

    def exists(self) -> bool:
        """True if the database file exists."""
        return Path(self._db_path).is_file()

    def _query(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Run one SELECT on a short-lived read-only connection.

        Args:
            sql: SQL text.
            params: Bound parameters.
        Returns:
            Rows; empty if the file is missing or the tables do not exist yet.
        """
        if not self.exists():
            return []
        try:
            conn = connect(self._db_path, read_only=True)
        except sqlite3.OperationalError:
            return []
        try:
            return conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    def list_runs(self, limit: int = 50) -> List[RunRecord]:
        """Runs, newest first.

        Args:
            limit: Maximum number of runs.
        Returns:
            RunRecords ordered by start time descending.
        """
        rows = self._query("SELECT run_id, started_ts, ended_ts, source_uri FROM runs ORDER BY started_ts DESC LIMIT ?", (limit,))
        return [RunRecord(r["run_id"], r["started_ts"], r["ended_ts"], r["source_uri"]) for r in rows]

    def latest_run_id(self) -> Optional[str]:
        """Id of the most recently started run, or None when there are no runs."""
        runs = self.list_runs(limit=1)
        return runs[0].run_id if runs else None

    def list_sessions(self, run_id: str, limit: int = 500) -> List[SessionRecord]:
        """Sessions of a run, newest first.

        Args:
            run_id: Run id.
            limit: Maximum number of sessions.
        Returns:
            SessionRecords ordered by first sighting descending.
        """
        rows = self._query(
            "SELECT * FROM sessions WHERE run_id = ? ORDER BY first_seen_ts DESC, session_no DESC LIMIT ?", (run_id, limit)
        )
        return [
            SessionRecord(
                r["run_id"], r["session_no"], r["aircraft_id"], r["first_seen_ts"], r["end_ts"], r["status"],
                r["closure_reason"], bool(r["partial_start"]), bool(r["partial_end"]),
            )
            for r in rows
        ]

    def list_intervals(self, run_id: str) -> List[IntervalRecord]:
        """All service intervals of a run (single query, grouped by the caller).

        Args:
            run_id: Run id.
        Returns:
            IntervalRecords ordered by session then start time.
        """
        rows = self._query("SELECT * FROM service_intervals WHERE run_id = ? ORDER BY session_no, start_ts", (run_id,))
        return [IntervalRecord(r["run_id"], r["session_no"], r["service"], r["start_ts"], r["end_ts"], r["end_reason"]) for r in rows]

    def list_events(self, run_id: str, session_no: Optional[int] = None, limit: int = 500) -> List[EventRecord]:
        """Events of a run (optionally of one session), newest first.

        Args:
            run_id: Run id.
            session_no: Restrict to this session when not None.
            limit: Maximum number of events.
        Returns:
            EventRecords with decoded payloads.
        """
        if session_no is None:
            rows = self._query("SELECT * FROM events WHERE run_id = ? ORDER BY ts DESC, event_id DESC LIMIT ?", (run_id, limit))
        else:
            rows = self._query(
                "SELECT * FROM events WHERE run_id = ? AND session_no = ? ORDER BY ts DESC, event_id DESC LIMIT ?",
                (run_id, session_no, limit),
            )
        return [
            EventRecord(
                r["event_id"], r["run_id"], r["ts"], r["event_type"], r["session_no"], r["aircraft_id"], r["service"],
                json.loads(r["payload_json"] or "{}"),
            )
            for r in rows
        ]

    def latest_event_ts(self, run_id: str) -> Optional[float]:
        """Timestamp of the newest event of a run (used as the stream clock for open sessions), or None.

        Args:
            run_id: Run id.
        """
        rows = self._query("SELECT MAX(ts) AS t FROM events WHERE run_id = ?", (run_id,))
        return rows[0]["t"] if rows and rows[0]["t"] is not None else None

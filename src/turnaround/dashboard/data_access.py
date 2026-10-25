"""Dashboard data shaping: records -> summaries -> pandas DataFrames (no Streamlit imports)."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from turnaround.analytics.summary import TurnaroundSummary, compute_session_summary
from turnaround.domain.records import EventRecord, IntervalRecord
from turnaround.domain.types import SERVICE_NAMES
from turnaround.storage.reader import AnalyticsReader


def format_duration(seconds: Optional[float]) -> str:
    """Format seconds as ``H:MM:SS`` (or ``M:SS`` below one hour).

    Args:
        seconds: Duration, or None.
    Returns:
        Formatted string, ``-`` for None.
    """
    if seconds is None:
        return "-"
    total = int(round(max(0.0, seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def to_datetime(ts: Optional[float]) -> Optional[datetime]:
    """Convert epoch seconds to a local naive datetime.

    Args:
        ts: Epoch seconds or None.
    Returns:
        datetime or None.
    """
    return datetime.fromtimestamp(ts) if ts is not None else None


def resolve_now(reader: AnalyticsReader, live_status: Optional[Mapping[str, Any]], run_id: str) -> float:
    """Pick the reference "now" (stream clock) used to close open intervals in analytics.

    Args:
        reader: Analytics reader.
        live_status: Parsed status.json or None.
        run_id: Selected run.
    Returns:
        ``status.stream_ts`` if the status belongs to the run; else the latest event timestamp of the run;
        else wall-clock time.
    """
    if live_status and live_status.get("run_id") == run_id and live_status.get("stream_ts"):
        return float(live_status["stream_ts"])
    latest = reader.latest_event_ts(run_id)
    return float(latest) if latest is not None else time.time()


def compute_summaries(reader: AnalyticsReader, run_id: str, now: float) -> List[TurnaroundSummary]:
    """Compute a TurnaroundSummary for every session of a run (two queries total).

    Args:
        reader: Analytics reader.
        run_id: Run id.
        now: Reference time for open sessions/intervals.
    Returns:
        Summaries ordered newest session first.
    """
    sessions = reader.list_sessions(run_id)
    by_session: Dict[int, List[IntervalRecord]] = {}
    for iv in reader.list_intervals(run_id):
        by_session.setdefault(iv.session_no, []).append(iv)
    return [compute_session_summary(s, by_session.get(s.session_no, []), now) for s in sessions]


def _minutes(seconds: Optional[float]) -> Optional[float]:
    """Seconds to minutes rounded to 1 decimal (None passes through).

    Args:
        seconds: Duration in seconds or None.
    """
    return None if seconds is None else round(seconds / 60.0, 1)


def summaries_to_frame(summaries: Sequence[TurnaroundSummary]) -> pd.DataFrame:
    """Tabular view of summaries for display.

    Args:
        summaries: Computed summaries.
    Returns:
        DataFrame with plain-language columns (durations in minutes; utilization in percent).
    Logic:
        One row per summary: session number, aircraft label, status, first-seen datetime, turnaround minutes,
        on-stand minutes, connected minutes per service (None if the service never occurred), idle minutes (None
        without a turnaround window), utilization percent and the reliability flag. Empty input gives an empty frame.
    """
    rows = []
    for s in summaries:
        row: Dict[str, Any] = {
            "Session": s.session_no,
            "Aircraft": f"A{s.aircraft_id}",
            "Status": s.status,
            "First seen": to_datetime(s.first_seen_ts),
            "Turnaround (min)": _minutes(s.turnaround_time_s),
            "On stand (min)": _minutes(s.session_duration_s),
        }
        for name in SERVICE_NAMES:
            row[f"{name.capitalize()} (min)"] = _minutes(s.services[name].total_s) if s.services[name].count else None
        row["Idle (min)"] = _minutes(s.idle_s) if s.turnaround_time_s is not None else None
        row["Utilization (%)"] = round(s.utilization * 100.0, 1) if s.utilization is not None else None
        row["Reliable"] = s.is_complete
        rows.append(row)
    return pd.DataFrame(rows)


def overview(summaries: Sequence[TurnaroundSummary]) -> Dict[str, Optional[float]]:
    """Headline KPIs over a run.

    Args:
        summaries: Computed summaries.
    Returns:
        Dict with sessions, open_sessions, reliable_sessions, mean_turnaround_s and median_turnaround_s
        (statistics use reliable sessions only; None when there are none).
    """
    reliable = [s.turnaround_time_s for s in summaries if s.is_complete and s.turnaround_time_s is not None]
    return {
        "sessions": float(len(summaries)),
        "open_sessions": float(sum(1 for s in summaries if s.status == "open")),
        "reliable_sessions": float(len(reliable)),
        "mean_turnaround_s": float(pd.Series(reliable).mean()) if reliable else None,
        "median_turnaround_s": float(pd.Series(reliable).median()) if reliable else None,
    }


def gantt_frame(intervals: Sequence[IntervalRecord], summary: TurnaroundSummary) -> pd.DataFrame:
    """Rows for the timeline chart of one session.

    Args:
        intervals: Stored intervals of the session.
        summary: The session's summary (supplies the end used for open intervals).
    Returns:
        DataFrame with columns service, start, end, duration_s, state (open/closed), end_reason.
    """
    rows = []
    for iv in intervals:
        end = iv.end_ts if iv.end_ts is not None else summary.reference_end_ts
        end = max(end, iv.start_ts)
        rows.append(
            {
                "service": iv.service,
                "start": to_datetime(iv.start_ts),
                "end": to_datetime(end),
                "duration_s": end - iv.start_ts,
                "state": "open" if iv.end_ts is None else "closed",
                "end_reason": iv.end_reason or "",
            }
        )
    return pd.DataFrame(rows, columns=["service", "start", "end", "duration_s", "state", "end_reason"])


def events_to_frame(events: Sequence[EventRecord]) -> pd.DataFrame:
    """Tabular view of events.

    Args:
        events: Event records.
    Returns:
        DataFrame with columns Time, Event, Session, Aircraft, Service, Details.
    """
    rows = [
        {
            "Time": to_datetime(e.ts),
            "Event": e.event_type,
            "Session": e.session_no,
            "Aircraft": f"A{e.aircraft_id}" if e.aircraft_id is not None else None,
            "Service": e.service,
            "Details": ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in e.payload.items()),
        }
        for e in events
    ]
    return pd.DataFrame(rows, columns=["Time", "Event", "Session", "Aircraft", "Service", "Details"])

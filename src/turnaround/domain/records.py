"""Read-model records returned by the storage reader and consumed by analytics and the dashboard."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RunRecord:
    """One pipeline run.

    Attributes:
        run_id: Unique run id.
        started_ts: Wall-clock start (epoch seconds).
        ended_ts: Wall-clock end, None while running or if the process was killed.
        source_uri: Source URL or path.
    """

    run_id: str
    started_ts: float
    ended_ts: Optional[float]
    source_uri: str


@dataclass(frozen=True)
class SessionRecord:
    """One stored turnaround session.

    Attributes:
        run_id: Run id.
        session_no: Session number within the run.
        aircraft_id: Pipeline aircraft id.
        first_seen_ts: Session start (stream clock).
        end_ts: Session end, None while open.
        status: ``open`` or ``closed``.
        closure_reason: EndReason value, None while open.
        partial_start: Aircraft already present at pipeline start.
        partial_end: Session cut by end of observation.
    """

    run_id: str
    session_no: int
    aircraft_id: int
    first_seen_ts: float
    end_ts: Optional[float]
    status: str
    closure_reason: Optional[str]
    partial_start: bool
    partial_end: bool


@dataclass(frozen=True)
class IntervalRecord:
    """One stored service interval.

    Attributes:
        run_id: Run id.
        session_no: Session number.
        service: Service name.
        start_ts: Interval start (stream clock).
        end_ts: Interval end, None while open.
        end_reason: EndReason value, None while open.
    """

    run_id: str
    session_no: int
    service: str
    start_ts: float
    end_ts: Optional[float]
    end_reason: Optional[str]


@dataclass(frozen=True)
class EventRecord:
    """One stored domain event.

    Attributes:
        event_id: Auto-increment id.
        run_id: Run id.
        ts: Effective event timestamp.
        event_type: EventType value.
        session_no: Session number or None.
        aircraft_id: Aircraft id or None.
        service: Service name or None.
        payload: Decoded payload JSON.
    """

    event_id: int
    run_id: str
    ts: float
    event_type: str
    session_no: Optional[int]
    aircraft_id: Optional[int]
    service: Optional[str]
    payload: Dict[str, Any]

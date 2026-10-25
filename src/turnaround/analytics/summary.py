"""Turnaround KPIs for one session computed from stored service intervals."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from turnaround.analytics.intervals import (
    Interval,
    clip_intervals,
    complement,
    concurrency_profile,
    merge_intervals,
    total_length,
)
from turnaround.domain.events import EndReason
from turnaround.domain.types import SERVICE_NAMES
from turnaround.domain.records import IntervalRecord, SessionRecord


@dataclass
class ServiceStats:
    """Aggregates for one service in one session.

    Attributes:
        service: Service name.
        total_s: Union length of the service's intervals.
        count: Number of stored intervals (re-connections increase it).
        first_start_ts: Earliest interval start, None if the service never occurred.
        last_end_ts: Latest interval end (open intervals use the reference end), None if never occurred.
        is_open: True if an interval is still open.
        censored: True if any interval ended because observation ended (shutdown/stream end) or is open.
    """

    service: str
    total_s: float = 0.0
    count: int = 0
    first_start_ts: Optional[float] = None
    last_end_ts: Optional[float] = None
    is_open: bool = False
    censored: bool = False


@dataclass
class TurnaroundSummary:
    """KPI bundle of one turnaround session.

    Attributes:
        run_id: Run the session belongs to.
        session_no: Session number within the run.
        aircraft_id: Aircraft id.
        status: ``open`` or ``closed``.
        first_seen_ts: Session start (first aircraft sighting).
        end_ts: Session end, None while open.
        reference_end_ts: End used for calculations (session end, or ``now`` while open).
        partial_start: Aircraft was already present at pipeline start.
        partial_end: Session was cut by end of observation.
        closure_reason: Reason the session ended, None while open.
        session_duration_s: reference_end - first_seen.
        window_start_ts: Earliest service start (None if no service occurred).
        window_end_ts: First pushback start if any, else latest service end.
        turnaround_time_s: window_end - window_start (None if no service occurred).
        arrival_to_first_service_s: window_start - first_seen.
        pushback_to_departure_s: reference_end - pushback start (None without pushback).
        busy_s: Time inside the window with at least one service connected.
        idle_s: Time inside the window with no service connected.
        utilization: busy_s / window length (None when the window is empty).
        parallel_s: Time inside the window with at least two services connected.
        max_concurrent: Maximum simultaneously connected services inside the window.
        idle_gaps: Idle intervals (start_ts, end_ts) inside the window.
        services: ServiceStats keyed by service name (all four services present).
        is_complete: Closed, pushback observed, and neither partial start nor partial end (KPIs reliable).
    """

    run_id: str
    session_no: int
    aircraft_id: int
    status: str
    first_seen_ts: float
    end_ts: Optional[float]
    reference_end_ts: float
    partial_start: bool
    partial_end: bool
    closure_reason: Optional[str]
    session_duration_s: float
    window_start_ts: Optional[float] = None
    window_end_ts: Optional[float] = None
    turnaround_time_s: Optional[float] = None
    arrival_to_first_service_s: Optional[float] = None
    pushback_to_departure_s: Optional[float] = None
    busy_s: float = 0.0
    idle_s: float = 0.0
    utilization: Optional[float] = None
    parallel_s: float = 0.0
    max_concurrent: int = 0
    idle_gaps: List[Interval] = field(default_factory=list)
    services: Dict[str, ServiceStats] = field(default_factory=dict)
    is_complete: bool = False


def compute_session_summary(session: SessionRecord, intervals: Sequence[IntervalRecord], now: float) -> TurnaroundSummary:
    """Compute all KPIs of one session.

    Args:
        session: Stored session row.
        intervals: Stored service intervals of that session (any order).
        now: Reference time for open sessions/intervals (stream clock, epoch seconds).
    Returns:
        TurnaroundSummary.
    Logic:
        1. ``ref_end`` = session end, or ``max(now, first_seen)`` while open. Open intervals end at ref_end;
           every interval is forced to end >= start.
        2. Per service: merge own intervals -> total_s, count, first start, last end, open/censored flags.
        3. Window: start = earliest service start over all services; end = start of the first pushback
           interval when pushback occurred, else the latest end; end is never before start.
        4. Busy = union of all intervals clipped to the window; idle gaps = window minus busy; utilization =
           busy / window length; parallel time and max concurrency via ``concurrency_profile`` on the
           per-service merged intervals clipped to the window.
        5. arrival_to_first_service and pushback_to_departure are derived from the session bounds.
        6. is_complete = closed and pushback seen and not partial_start and not partial_end.
    """
    ref_end = session.end_ts if session.end_ts is not None else max(now, session.first_seen_ts)
    per_service: Dict[str, List[Interval]] = {name: [] for name in SERVICE_NAMES}
    stats: Dict[str, ServiceStats] = {name: ServiceStats(service=name) for name in SERVICE_NAMES}
    censoring = {EndReason.STREAM_END.value, EndReason.SHUTDOWN.value}
    for rec in intervals:
        if rec.service not in per_service:
            continue
        end = rec.end_ts if rec.end_ts is not None else ref_end
        end = max(end, rec.start_ts)
        per_service[rec.service].append((rec.start_ts, end))
        st = stats[rec.service]
        st.count += 1
        st.first_start_ts = rec.start_ts if st.first_start_ts is None else min(st.first_start_ts, rec.start_ts)
        st.last_end_ts = end if st.last_end_ts is None else max(st.last_end_ts, end)
        if rec.end_ts is None:
            st.is_open = True
            st.censored = True
        elif rec.end_reason in censoring:
            st.censored = True

    merged = {name: merge_intervals(ivs) for name, ivs in per_service.items()}
    for name, st in stats.items():
        st.total_s = total_length(merged[name])

    summary = TurnaroundSummary(
        run_id=session.run_id,
        session_no=session.session_no,
        aircraft_id=session.aircraft_id,
        status=session.status,
        first_seen_ts=session.first_seen_ts,
        end_ts=session.end_ts,
        reference_end_ts=ref_end,
        partial_start=session.partial_start,
        partial_end=session.partial_end,
        closure_reason=session.closure_reason,
        session_duration_s=max(0.0, ref_end - session.first_seen_ts),
        services=stats,
    )

    all_ivs = [iv for ivs in merged.values() for iv in ivs]
    if not all_ivs:
        return summary

    window_start = min(s for s, _ in all_ivs)
    pushback = merged["pushback"]
    window_end = pushback[0][0] if pushback else max(e for _, e in all_ivs)
    window_end = max(window_end, window_start)
    summary.window_start_ts = window_start
    summary.window_end_ts = window_end
    summary.turnaround_time_s = window_end - window_start
    summary.arrival_to_first_service_s = window_start - session.first_seen_ts
    if pushback:
        summary.pushback_to_departure_s = max(0.0, ref_end - pushback[0][0])

    busy = merge_intervals(clip_intervals(all_ivs, window_start, window_end))
    summary.busy_s = total_length(busy)
    summary.idle_gaps = complement((window_start, window_end), busy)
    summary.idle_s = total_length(summary.idle_gaps)
    length = window_end - window_start
    summary.utilization = summary.busy_s / length if length > 0 else None
    clipped_groups = [clip_intervals(ivs, window_start, window_end) for ivs in merged.values()]
    summary.parallel_s, summary.max_concurrent = concurrency_profile(clipped_groups)
    summary.is_complete = session.status == "closed" and bool(pushback) and not session.partial_start and not session.partial_end
    return summary


def window_bounds(summary: TurnaroundSummary) -> Optional[Tuple[float, float]]:
    """Service window of a summary.

    Args:
        summary: Computed summary.
    Returns:
        (start, end) or None when no service occurred.
    """
    if summary.window_start_ts is None or summary.window_end_ts is None:
        return None
    return summary.window_start_ts, summary.window_end_ts

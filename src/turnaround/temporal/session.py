"""One turnaround session: per-service hysteresis filters that emit domain events."""
from __future__ import annotations

from typing import Dict, List, Mapping

from turnaround.config.schema import ServiceRuleConfig
from turnaround.domain.events import DomainEvent, EndReason, EventType
from turnaround.domain.types import ALL_SERVICES, ServiceType
from turnaround.temporal.hysteresis import HysteresisFilter, Transition, TransitionKind


class TurnaroundSession:
    """State of one aircraft's turnaround (from first confirmed sighting to departure/shutdown)."""

    def __init__(
        self,
        session_no: int,
        aircraft_id: int,
        first_seen_ts: float,
        partial_start: bool,
        rules: Mapping[ServiceType, ServiceRuleConfig],
    ) -> None:
        """Create a session with one HysteresisFilter per service.

        Args:
            session_no: Session number within the run (1-based).
            aircraft_id: Aircraft the session belongs to.
            first_seen_ts: Timestamp of the first sighting (session start).
            partial_start: True if the aircraft was already present when the pipeline started.
            rules: Hysteresis rule for every ServiceType.
        """
        self.session_no = session_no
        self.aircraft_id = aircraft_id
        self.first_seen_ts = first_seen_ts
        self.partial_start = partial_start
        self._rules = dict(rules)
        self._filters: Dict[ServiceType, HysteresisFilter] = {svc: HysteresisFilter(rules[svc]) for svc in ALL_SERVICES}
        self._closed = False

    @property
    def closed(self) -> bool:
        """True after ``close`` was called."""
        return self._closed

    def start_event(self) -> DomainEvent:
        """The SESSION_STARTED event (ts = first sighting, payload partial_start)."""
        return DomainEvent(
            EventType.SESSION_STARTED,
            ts=self.first_seen_ts,
            session_no=self.session_no,
            aircraft_id=self.aircraft_id,
            payload={"partial_start": self.partial_start},
        )

    def active_services(self) -> List[ServiceType]:
        """Services currently debounced as connected, in canonical order."""
        return [svc for svc in ALL_SERVICES if self._filters[svc].active]

    def update(self, ts: float, dt: float, presence: Mapping[ServiceType, bool]) -> List[DomainEvent]:
        """Feed one frame of evidence to every service filter.

        Args:
            ts: Frame timestamp.
            dt: Observed seconds since the previous processed frame.
            presence: Boolean evidence per service for this aircraft.
        Returns:
            Events produced by transitions in this frame (possibly empty).
        Logic:
            For each service call ``HysteresisFilter.update``. ACTIVATED -> SERVICE_STARTED at the
            back-dated start. DEACTIVATED -> if the interval is shorter than ``min_interval_s`` emit
            SERVICE_DISCARDED (false trigger retracted) else SERVICE_ENDED(end_reason=released).
        """
        events: List[DomainEvent] = []
        for svc in ALL_SERVICES:
            transition = self._filters[svc].update(ts, dt, bool(presence.get(svc, False)))
            if transition is None:
                continue
            if transition.kind is TransitionKind.ACTIVATED:
                events.append(self._service_event(EventType.SERVICE_STARTED, svc, transition.effective_ts, {"decided_at": transition.decided_ts}))
                continue
            duration = transition.effective_ts - transition.start_ts
            payload = {"duration_s": duration, "decided_at": transition.decided_ts}
            if duration < self._rules[svc].min_interval_s:
                payload["reason"] = "shorter_than_min_interval"
                events.append(self._service_event(EventType.SERVICE_DISCARDED, svc, transition.effective_ts, payload))
            else:
                payload["end_reason"] = EndReason.RELEASED.value
                events.append(self._service_event(EventType.SERVICE_ENDED, svc, transition.effective_ts, payload))
        return events

    def close(self, end_ts: float, reason: EndReason, partial_end: bool = False) -> List[DomainEvent]:
        """Close the session, flushing every active service interval.

        Args:
            end_ts: Session end timestamp; service intervals are clamped to it.
            reason: Why the session ends (AIRCRAFT_DEPARTED, STREAM_END or SHUTDOWN).
            partial_end: True when the session was cut by the end of observation rather than a departure.
        Returns:
            SERVICE_ENDED events for intervals that were active, then one SESSION_ENDED event. Empty if
            already closed.
        Logic:
            Active filters are force-closed; interval end = ``max(start, min(last_positive_ts, end_ts))``.
            Forced closes are never discarded, even if short (the data may just be cut by shutdown).
        """
        if self._closed:
            return []
        self._closed = True
        events: List[DomainEvent] = []
        for svc in ALL_SERVICES:
            transition = self._filters[svc].force_close(end_ts)
            if transition is None:
                continue
            effective_end = max(transition.start_ts, min(transition.effective_ts, end_ts))
            events.append(
                self._service_event(
                    EventType.SERVICE_ENDED,
                    svc,
                    effective_end,
                    {"duration_s": effective_end - transition.start_ts, "decided_at": end_ts, "end_reason": reason.value},
                )
            )
        events.append(
            DomainEvent(
                EventType.SESSION_ENDED,
                ts=end_ts,
                session_no=self.session_no,
                aircraft_id=self.aircraft_id,
                payload={"reason": reason.value, "partial_end": partial_end},
            )
        )
        return events

    def _service_event(self, event_type: EventType, service: ServiceType, ts: float, payload: Dict) -> DomainEvent:
        """Build a service-level event bound to this session.

        Args:
            event_type: SERVICE_STARTED, SERVICE_ENDED or SERVICE_DISCARDED.
            service: Service concerned.
            ts: Effective timestamp.
            payload: Event payload.
        Returns:
            DomainEvent carrying this session's number and aircraft id.
        """
        return DomainEvent(event_type, ts=ts, session_no=self.session_no, aircraft_id=self.aircraft_id, service=service, payload=payload)

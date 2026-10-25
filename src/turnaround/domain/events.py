"""Domain events emitted by the temporal engine and consumed by sinks.

Events are the only way the temporal layer communicates state changes to the
outside world (storage, logging, future alerting). They are immutable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional

from turnaround.domain.types import ServiceType


class EventType(str, Enum):
    """Kinds of domain events.

    Attributes:
        SESSION_STARTED: A confirmed aircraft opened a turnaround session. ts = first sighting.
        SESSION_ENDED: A session closed. ts = session end. payload: reason, partial_end.
        SERVICE_STARTED: A service became connected. ts = back-dated first positive evidence.
        SERVICE_ENDED: A service interval closed. ts = last positive evidence (clamped to session end).
        SERVICE_DISCARDED: A too-short interval was retracted as a false trigger.
        SCENE_CHANGED: The scene guard detected a camera view change; analysis paused.
        SCENE_RESTORED: The original view returned; analysis resumed.
    """

    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    SERVICE_STARTED = "service_started"
    SERVICE_ENDED = "service_ended"
    SERVICE_DISCARDED = "service_discarded"
    SCENE_CHANGED = "scene_changed"
    SCENE_RESTORED = "scene_restored"


class EndReason(str, Enum):
    """Why an interval or a session ended.

    Attributes:
        RELEASED: Interval ended because evidence vanished for the grace period (normal).
        AIRCRAFT_DEPARTED: Aircraft was lost beyond the lost timeout, closing the session.
        STREAM_END: The finite source (video file) ended.
        SHUTDOWN: The pipeline was stopped by the operator or a signal.
    """

    RELEASED = "released"
    AIRCRAFT_DEPARTED = "aircraft_departed"
    STREAM_END = "stream_end"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class DomainEvent:
    """One immutable state-change notification.

    Attributes:
        event_type: Kind of event.
        ts: Effective stream timestamp (epoch seconds) the event refers to.
        session_no: Turnaround session number within the run, None for scene-level events.
        aircraft_id: Pipeline aircraft id, None for scene-level events.
        service: Service concerned, None for session/scene events.
        payload: Extra JSON-serialisable details (decided_at, duration_s, reason, ...).
    """

    event_type: EventType
    ts: float
    session_no: Optional[int] = None
    aircraft_id: Optional[int] = None
    service: Optional[ServiceType] = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly dictionary form (enums replaced by their string values)."""
        return {
            "event_type": self.event_type.value,
            "ts": self.ts,
            "session_no": self.session_no,
            "aircraft_id": self.aircraft_id,
            "service": self.service.value if self.service else None,
            "payload": dict(self.payload),
        }

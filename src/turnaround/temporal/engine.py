"""TurnaroundEngine: pure (I/O-free) core turning per-frame detections into domain events."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from turnaround.config.schema import TemporalConfig
from turnaround.domain.events import DomainEvent, EndReason
from turnaround.domain.types import ALL_SERVICES, AircraftSnapshot, Detection, DetectionKind, ServiceType
from turnaround.spatial.association import AssociationResult, EquipmentAssociator
from turnaround.spatial.roi import RoiSet
from turnaround.temporal.aircraft_registry import AircraftRegistry
from turnaround.temporal.session import TurnaroundSession

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Everything the engine produced for one frame (consumed by sinks and the annotator).

    Attributes:
        ts: Frame timestamp.
        dt: Observed seconds credited for this frame (capped by max_gap_s).
        aircraft: Confirmed aircraft snapshots (observed and coasting).
        association: Accepted/rejected equipment detections.
        events: Domain events emitted in this frame.
        session_by_aircraft: aircraft_id -> session_no for open sessions.
        active_services: aircraft_id -> services currently debounced as connected.
    """

    ts: float
    dt: float
    aircraft: List[AircraftSnapshot] = field(default_factory=list)
    association: AssociationResult = field(default_factory=AssociationResult)
    events: List[DomainEvent] = field(default_factory=list)
    session_by_aircraft: Dict[int, int] = field(default_factory=dict)
    active_services: Dict[int, List[ServiceType]] = field(default_factory=dict)


class TurnaroundEngine:
    """Orchestrates registry -> association -> per-session hysteresis for each frame.

    The engine owns no I/O. It receives detections and returns events, which makes it fully
    unit-testable with synthetic data and reusable behind any frontend.
    """

    def __init__(self, cfg: TemporalConfig, registry: AircraftRegistry, associator: EquipmentAssociator, roi: RoiSet) -> None:
        """Wire the engine's collaborators.

        Args:
            cfg: Temporal configuration (gap cap, partial-start window, per-service rules).
            registry: Aircraft identity registry.
            associator: Equipment associator.
            roi: ROI set (bound to the frame size on every ``process`` call).
        """
        self._cfg = cfg
        self._registry = registry
        self._associator = associator
        self._roi = roi
        self._rules = {svc: cfg.rule_for(svc.value) for svc in ALL_SERVICES}
        self._sessions: Dict[int, TurnaroundSession] = {}
        self._next_session_no = 1
        self._first_ts: Optional[float] = None
        self._last_ts: Optional[float] = None

    @property
    def last_ts(self) -> Optional[float]:
        """Timestamp of the last processed frame (None before the first)."""
        return self._last_ts

    def open_sessions(self) -> List[TurnaroundSession]:
        """Currently open sessions ordered by session number."""
        return sorted(self._sessions.values(), key=lambda s: s.session_no)

    def process(self, ts: float, detections: Sequence[Detection], frame_size: Tuple[int, int]) -> FrameResult:
        """Process one frame of detections.

        Args:
            ts: Frame timestamp (stream clock, monotonic non-decreasing).
            detections: Mapped detections of the frame (aircraft and equipment).
            frame_size: (width, height) of the frame, used to bind ROI polygons.
        Returns:
            FrameResult with events and renderable state.
        Logic:
            1. Compute ``dt = clamp(ts - last_ts, 0, max_gap_s)`` (0 for the first frame).
            2. Bind ROI to the frame size; split detections into ROI-filtered aircraft and equipment.
            3. ``registry.update``; close sessions of departed aircraft (end = last sighting,
               reason AIRCRAFT_DEPARTED); open sessions for newly confirmed aircraft
               (partial_start if first seen within ``partial_start_window_s`` of the first frame).
            4. ``associator.associate`` against confirmed snapshots (observed + coasting).
            5. Feed each open session with its per-service presence and collect events.
        """
        dt = 0.0 if self._last_ts is None else min(max(ts - self._last_ts, 0.0), self._cfg.max_gap_s)
        if self._first_ts is None:
            self._first_ts = ts
        self._last_ts = ts if self._last_ts is None else max(self._last_ts, ts)

        self._roi.bind(*frame_size)
        aircraft_dets = [d for d in detections if d.kind is DetectionKind.AIRCRAFT and self._roi.allows_aircraft(d.box)]
        equipment_dets = [d for d in detections if d.kind is DetectionKind.EQUIPMENT]

        events: List[DomainEvent] = []
        update = self._registry.update(ts, dt, aircraft_dets)

        for gone in update.departed:
            session = self._sessions.pop(gone.aircraft_id, None)
            if session is not None:
                events.extend(session.close(gone.last_seen_ts, EndReason.AIRCRAFT_DEPARTED, partial_end=False))

        for snap in update.newly_confirmed:
            partial = (snap.first_seen_ts - self._first_ts) <= self._cfg.partial_start_window_s
            session = TurnaroundSession(self._next_session_no, snap.aircraft_id, snap.first_seen_ts, partial, self._rules)
            self._next_session_no += 1
            self._sessions[snap.aircraft_id] = session
            events.append(session.start_event())

        association = self._associator.associate(equipment_dets, update.snapshots)

        for aircraft_id, session in self._sessions.items():
            events.extend(session.update(ts, dt, association.presence_for(aircraft_id)))

        return FrameResult(
            ts=ts,
            dt=dt,
            aircraft=update.snapshots,
            association=association,
            events=events,
            session_by_aircraft={aid: s.session_no for aid, s in self._sessions.items()},
            active_services={aid: s.active_services() for aid, s in self._sessions.items()},
        )

    def flush(self, ts: float, reason: EndReason) -> List[DomainEvent]:
        """Close every open session (end of stream or shutdown).

        Args:
            ts: End timestamp (typically the last frame timestamp).
            reason: STREAM_END or SHUTDOWN.
        Returns:
            SERVICE_ENDED / SESSION_ENDED events for all open sessions (partial_end=True).
        """
        events: List[DomainEvent] = []
        for aircraft_id in sorted(self._sessions):
            events.extend(self._sessions[aircraft_id].close(ts, reason, partial_end=True))
        self._sessions.clear()
        return events

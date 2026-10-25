"""Shared builders for tests (no pytest imports so they can be reused anywhere)."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from turnaround.config.schema import AircraftConfig, AssociationConfig, RoiConfig, ServiceRuleConfig, TemporalConfig
from turnaround.domain.events import DomainEvent
from turnaround.domain.types import BoundingBox, Detection, DetectionKind, ServiceType
from turnaround.spatial.association import EquipmentAssociator
from turnaround.spatial.roi import RoiSet
from turnaround.temporal.aircraft_registry import AircraftRegistry
from turnaround.temporal.engine import TurnaroundEngine

AIRCRAFT_BOX = BoundingBox(100, 100, 500, 300)
EQUIP_BOX = BoundingBox(400, 150, 480, 250)  # lies on the aircraft
FAR_BOX = BoundingBox(520, 320, 600, 350)  # away from the aircraft
FRAME_SIZE = (640, 360)
BASE_TS = 1_000_000.0
DT = 0.2  # 5 FPS


def aircraft(box: BoundingBox = AIRCRAFT_BOX, conf: float = 0.9, track_id: Optional[int] = 1) -> Detection:
    """Aircraft detection."""
    return Detection(box, conf, 0, "aircraft", track_id, DetectionKind.AIRCRAFT, None)


def equipment(service: ServiceType, box: BoundingBox = EQUIP_BOX, conf: float = 0.9, track_id: Optional[int] = None) -> Detection:
    """Equipment detection."""
    return Detection(box, conf, 1, f"{service.value}_connected", track_id, DetectionKind.EQUIPMENT, service)


def make_engine(
    aircraft_cfg: Optional[AircraftConfig] = None,
    temporal: Optional[TemporalConfig] = None,
    roi_cfg: Optional[RoiConfig] = None,
    assoc: Optional[AssociationConfig] = None,
) -> TurnaroundEngine:
    """Engine with test-friendly defaults (short aircraft lost timeout)."""
    aircraft_cfg = aircraft_cfg or AircraftConfig(lost_timeout_s=20.0)
    temporal = temporal or TemporalConfig(defaults=ServiceRuleConfig(on_window_s=3.0, on_min_hits=4, off_grace_s=8.0, min_interval_s=5.0))
    roi = RoiSet(roi_cfg or RoiConfig())
    return TurnaroundEngine(temporal, AircraftRegistry(aircraft_cfg), EquipmentAssociator(assoc or AssociationConfig(), roi), roi)


def run_script(
    engine: TurnaroundEngine,
    duration_s: float,
    frame_fn: Callable[[float], Sequence[Detection]],
    skip: Callable[[float], bool] = lambda t: False,
) -> List[DomainEvent]:
    """Drive the engine at 5 FPS and collect every event.

    Args:
        engine: Engine under test.
        duration_s: Simulated seconds.
        frame_fn: Maps relative time t -> detections for that frame.
        skip: Returns True for instants at which no frame is delivered (stream stall).
    """
    events: List[DomainEvent] = []
    n = int(round(duration_s / DT))
    for i in range(n):
        t = i * DT
        if skip(t):
            continue
        events.extend(engine.process(BASE_TS + t, frame_fn(t), FRAME_SIZE).events)
    return events


def by_type(events: Sequence[DomainEvent], event_type, service: Optional[ServiceType] = None) -> List[DomainEvent]:
    """Filter events by type (and service)."""
    return [e for e in events if e.event_type is event_type and (service is None or e.service is service)]

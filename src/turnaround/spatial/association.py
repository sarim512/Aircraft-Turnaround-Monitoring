"""Validates equipment detections (confidence, ROI, aircraft overlap) and assigns them to aircraft."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Sequence, Tuple

from turnaround.config.schema import AssociationConfig
from turnaround.domain.types import ALL_SERVICES, AircraftSnapshot, Detection, ServiceType
from turnaround.spatial.roi import RoiSet


class RejectionReason(str, Enum):
    """Why an equipment detection did not count as evidence.

    Attributes:
        LOW_CONFIDENCE: Below the per-service confidence floor.
        OUTSIDE_ANALYSIS_ZONE: Anchor outside the analysis polygon.
        OUTSIDE_SERVICE_ZONE: Anchor outside the service-specific polygon.
        NO_AIRCRAFT_CONTEXT: No confirmed aircraft exists in this frame.
        NOT_ON_AIRCRAFT: Overlap with every (enlarged) aircraft box is below the threshold.
    """

    LOW_CONFIDENCE = "low_confidence"
    OUTSIDE_ANALYSIS_ZONE = "outside_analysis_zone"
    OUTSIDE_SERVICE_ZONE = "outside_service_zone"
    NO_AIRCRAFT_CONTEXT = "no_aircraft_context"
    NOT_ON_AIRCRAFT = "not_on_aircraft"


@dataclass(frozen=True)
class AcceptedEquipment:
    """An equipment detection accepted as evidence for one aircraft.

    Attributes:
        detection: The detection.
        service: Its service type.
        aircraft_id: Aircraft it was assigned to.
        overlap: Fraction of the equipment box inside the enlarged aircraft box.
    """

    detection: Detection
    service: ServiceType
    aircraft_id: int
    overlap: float


@dataclass(frozen=True)
class RejectedEquipment:
    """An equipment detection that was filtered out.

    Attributes:
        detection: The detection.
        reason: Why it was rejected.
    """

    detection: Detection
    reason: RejectionReason


@dataclass
class AssociationResult:
    """Outcome of associating one frame's equipment detections.

    Attributes:
        accepted: Detections accepted as evidence.
        rejected: Detections filtered out, with reasons.
    """

    accepted: List[AcceptedEquipment] = field(default_factory=list)
    rejected: List[RejectedEquipment] = field(default_factory=list)

    def presence_for(self, aircraft_id: int) -> Dict[ServiceType, bool]:
        """Per-service boolean evidence for one aircraft in this frame.

        Args:
            aircraft_id: Aircraft id.
        Returns:
            Dict with an entry for every ServiceType (True if at least one accepted detection).
        """
        present = {svc: False for svc in ALL_SERVICES}
        for acc in self.accepted:
            if acc.aircraft_id == aircraft_id:
                present[acc.service] = True
        return present


class EquipmentAssociator:
    """Turns raw equipment detections into per-aircraft per-service evidence."""

    def __init__(self, cfg: AssociationConfig, roi: RoiSet) -> None:
        """Store thresholds and the ROI set.

        Args:
            cfg: Association configuration.
            roi: Bound-per-frame ROI set shared with the engine.
        """
        self._cfg = cfg
        self._roi = roi

    def associate(self, equipment: Sequence[Detection], aircraft: Sequence[AircraftSnapshot]) -> AssociationResult:
        """Validate and assign every equipment detection of the frame.

        Args:
            equipment: Equipment detections (``service`` not None); others are ignored.
            aircraft: Confirmed aircraft snapshots (observed or coasting) for this frame.
        Returns:
            AssociationResult with accepted and rejected detections.
        Logic:
            For each detection, in this order: (1) confidence >= per-service floor, (2) anchor inside the
            analysis zone, (3) anchor inside the service zone, (4) at least one aircraft exists, (5) for
            every aircraft compute ``intersection(det_box, aircraft_box.expanded(margin)) / det_area`` and
            take the maximum (ties -> lowest aircraft id); reject if below ``min_equipment_overlap``.
            The first failing check decides the RejectionReason.
        """
        result = AssociationResult()
        for det in equipment:
            service = det.service
            if service is None:
                continue
            if det.confidence < self._cfg.min_confidence[service.value]:
                result.rejected.append(RejectedEquipment(det, RejectionReason.LOW_CONFIDENCE))
                continue
            if not self._roi.allows_equipment(det.box):
                result.rejected.append(RejectedEquipment(det, RejectionReason.OUTSIDE_ANALYSIS_ZONE))
                continue
            if not self._roi.allows_service(service, det.box):
                result.rejected.append(RejectedEquipment(det, RejectionReason.OUTSIDE_SERVICE_ZONE))
                continue
            if not aircraft:
                result.rejected.append(RejectedEquipment(det, RejectionReason.NO_AIRCRAFT_CONTEXT))
                continue
            best_id, best_overlap = self._best_aircraft(det, aircraft)
            if best_overlap < self._cfg.min_equipment_overlap:
                result.rejected.append(RejectedEquipment(det, RejectionReason.NOT_ON_AIRCRAFT))
                continue
            result.accepted.append(AcceptedEquipment(det, service, best_id, best_overlap))
        return result

    def _best_aircraft(self, det: Detection, aircraft: Sequence[AircraftSnapshot]) -> Tuple[int, float]:
        """Find the aircraft with the largest overlap fraction.

        Args:
            det: Equipment detection.
            aircraft: Non-empty aircraft snapshots.
        Returns:
            (aircraft_id, overlap fraction); overlap is 0.0 for a zero-area detection box.
        """
        area = det.box.area
        best_id = -1
        best = -1.0
        for ac in sorted(aircraft, key=lambda a: a.aircraft_id):
            inter = det.box.intersection_area(ac.box.expanded(self._cfg.aircraft_box_margin))
            ratio = inter / area if area > 0 else 0.0
            if ratio > best:
                best, best_id = ratio, ac.aircraft_id
        return best_id, best

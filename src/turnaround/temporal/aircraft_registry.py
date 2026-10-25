"""Stable aircraft identities across tracker id changes, occlusion and missed detections."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from turnaround.config.schema import AircraftConfig
from turnaround.domain.types import AircraftSnapshot, BoundingBox, Detection

logger = logging.getLogger(__name__)


@dataclass
class _Slot:
    """Mutable bookkeeping for one aircraft identity.

    Attributes:
        aircraft_id: Pipeline aircraft id.
        first_seen_ts: Stream timestamp of the first detection.
        last_seen_ts: Stream timestamp of the latest detection.
        box: Latest observed box.
        confidence: Latest observed confidence.
        hits: Number of frames in which the aircraft was detected.
        confirmed: True once hits >= confirm_hits.
        unseen_s: Observed seconds since the last detection.
        observed: Whether it was detected in the current frame.
        raw_track_id: Tracker id of the latest detection.
    """

    aircraft_id: int
    first_seen_ts: float
    last_seen_ts: float
    box: BoundingBox
    confidence: float
    hits: int = 0
    confirmed: bool = False
    unseen_s: float = 0.0
    observed: bool = False
    raw_track_id: Optional[int] = None


@dataclass(frozen=True)
class DepartedAircraft:
    """A confirmed aircraft that has been unseen longer than ``lost_timeout_s``.

    Attributes:
        aircraft_id: Aircraft id.
        first_seen_ts: First detection timestamp.
        last_seen_ts: Last detection timestamp (used as the session end).
    """

    aircraft_id: int
    first_seen_ts: float
    last_seen_ts: float


@dataclass
class RegistryUpdate:
    """Result of one registry update.

    Attributes:
        snapshots: Confirmed aircraft (observed or coasting) for this frame.
        newly_confirmed: Aircraft that crossed the confirmation threshold in this frame.
        departed: Confirmed aircraft that expired in this frame.
    """

    snapshots: List[AircraftSnapshot] = field(default_factory=list)
    newly_confirmed: List[AircraftSnapshot] = field(default_factory=list)
    departed: List[DepartedAircraft] = field(default_factory=list)


class AircraftRegistry:
    """Maintains aircraft identities ("slots") independent of raw ByteTrack ids.

    Matching per frame: (A) a detection whose tracker id is already bound to a slot continues that
    slot; (B) remaining detections are re-attached to remaining slots by greedy IoU matching
    (handles tracker id switches after occlusion); (C) anything left starts a tentative slot.
    Slots not matched coast on their last box until their observed-unseen time exceeds a timeout.
    """

    def __init__(self, cfg: AircraftConfig) -> None:
        """Create an empty registry.

        Args:
            cfg: Aircraft identity configuration.
        """
        self._cfg = cfg
        self._slots: Dict[int, _Slot] = {}
        self._track_to_slot: Dict[int, int] = {}
        self._next_id = 1

    def update(self, ts: float, dt: float, detections: Sequence[Detection]) -> RegistryUpdate:
        """Advance the registry by one frame.

        Args:
            ts: Stream timestamp of the frame.
            dt: Observed seconds since the previous processed frame (already capped by the engine).
            detections: Aircraft detections of the frame (ROI-filtered).
        Returns:
            RegistryUpdate with confirmed snapshots, newly confirmed and departed aircraft.
        Logic:
            1. ``_prepare``: drop low-confidence boxes and near-duplicate boxes.
            2. Phase A: bind by tracker id when the id maps to a live slot not yet matched.
            3. Phase B: for unmatched detections x unmatched slots, sort pairs by IoU descending and
               greedily match while IoU >= reid_iou_threshold; rebind the tracker id to the slot.
            4. Phase C: unmatched detections become new tentative slots.
            5. Update matched slots (box, confidence, last_seen_ts, hits, unseen_s=0, observed=True) and
               confirm those reaching confirm_hits.
            6. Unmatched slots: unseen_s += dt; tentative slots expire after tentative_timeout_s
               (dropped silently), confirmed slots after lost_timeout_s (reported as departed).
            7. Snapshots are produced only for confirmed slots.
        """
        cfg = self._cfg
        dets = self._prepare(detections)
        for slot in self._slots.values():
            slot.observed = False

        matched: Dict[int, Detection] = {}
        unmatched: List[Detection] = []
        for det in dets:
            sid = self._track_to_slot.get(det.track_id) if det.track_id is not None else None
            if sid is not None and sid in self._slots and sid not in matched:
                matched[sid] = det
            else:
                unmatched.append(det)

        free_slots = [s for sid, s in self._slots.items() if sid not in matched]
        pairs = sorted(
            ((det.box.iou(slot.box), di, slot.aircraft_id) for di, det in enumerate(unmatched) for slot in free_slots),
            reverse=True,
        )
        used_dets: set = set()
        used_slots: set = set()
        for iou, di, sid in pairs:
            if iou < cfg.reid_iou_threshold:
                break
            if di in used_dets or sid in used_slots:
                continue
            used_dets.add(di)
            used_slots.add(sid)
            matched[sid] = unmatched[di]
            if unmatched[di].track_id is not None:
                self._track_to_slot[unmatched[di].track_id] = sid

        for di, det in enumerate(unmatched):
            if di in used_dets:
                continue
            slot = _Slot(self._next_id, ts, ts, det.box, det.confidence)
            self._slots[slot.aircraft_id] = slot
            self._next_id += 1
            matched[slot.aircraft_id] = det
            if det.track_id is not None:
                self._track_to_slot[det.track_id] = slot.aircraft_id

        update = RegistryUpdate()
        for sid, det in matched.items():
            slot = self._slots[sid]
            slot.box = det.box
            slot.confidence = det.confidence
            slot.last_seen_ts = ts
            slot.unseen_s = 0.0
            slot.observed = True
            slot.raw_track_id = det.track_id
            slot.hits += 1
            if not slot.confirmed and slot.hits >= cfg.confirm_hits:
                slot.confirmed = True
                update.newly_confirmed.append(self._snapshot(slot))

        for sid in list(self._slots):
            if sid in matched:
                continue
            slot = self._slots[sid]
            slot.unseen_s += dt
            if not slot.confirmed and slot.unseen_s >= cfg.tentative_timeout_s:
                self._drop(sid)
            elif slot.confirmed and slot.unseen_s >= cfg.lost_timeout_s:
                update.departed.append(DepartedAircraft(sid, slot.first_seen_ts, slot.last_seen_ts))
                self._drop(sid)

        update.snapshots = [self._snapshot(s) for s in self._slots.values() if s.confirmed]
        return update

    def confirmed_ids(self) -> List[int]:
        """Ids of all currently confirmed (observed or coasting) aircraft, ascending."""
        return sorted(sid for sid, slot in self._slots.items() if slot.confirmed)

    def _prepare(self, detections: Sequence[Detection]) -> List[Detection]:
        """Filter and de-duplicate aircraft detections.

        Args:
            detections: Raw aircraft detections.
        Returns:
            Detections with confidence >= min_confidence, sorted by confidence descending, where any
            box overlapping an already kept box with IoU >= duplicate_iou_threshold is removed.
        """
        kept: List[Detection] = []
        for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
            if det.confidence < self._cfg.min_confidence:
                continue
            if any(det.box.iou(k.box) >= self._cfg.duplicate_iou_threshold for k in kept):
                continue
            kept.append(det)
        return kept

    def _drop(self, sid: int) -> None:
        """Remove a slot and every tracker-id binding that points to it.

        Args:
            sid: Aircraft id to remove.
        """
        self._slots.pop(sid, None)
        for track_id in [t for t, s in self._track_to_slot.items() if s == sid]:
            del self._track_to_slot[track_id]

    @staticmethod
    def _snapshot(slot: _Slot) -> AircraftSnapshot:
        """Freeze a slot into an immutable AircraftSnapshot.

        Args:
            slot: Slot to convert.
        Returns:
            Snapshot with ``observed`` taken from the slot.
        """
        return AircraftSnapshot(
            aircraft_id=slot.aircraft_id,
            box=slot.box,
            confidence=slot.confidence,
            observed=slot.observed,
            first_seen_ts=slot.first_seen_ts,
            last_seen_ts=slot.last_seen_ts,
            raw_track_id=slot.raw_track_id,
        )

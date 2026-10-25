"""Core immutable value types shared by every pipeline stage.

This module performs no I/O and imports no other ``turnaround`` package. Every
other package may import from it. numpy is used only to type the frame array.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np

ANCHOR_MODES: Tuple[str, ...] = ("center", "bottom_center")


class ServiceType(str, Enum):
    """Ground-service equipment categories tracked per aircraft.

    Attributes:
        AEROBRIDGE: Passenger boarding bridge connected to the aircraft.
        CATERING: Catering truck connected to the aircraft.
        CARGO: Cargo / baggage loader connected to the aircraft.
        PUSHBACK: Pushback tug connected to the aircraft.
    """

    AEROBRIDGE = "aerobridge"
    CATERING = "catering"
    CARGO = "cargo"
    PUSHBACK = "pushback"


ALL_SERVICES: Tuple[ServiceType, ...] = tuple(ServiceType)
SERVICE_NAMES: Tuple[str, ...] = tuple(s.value for s in ServiceType)


class DetectionKind(str, Enum):
    """Semantic role of a detection after class mapping.

    Attributes:
        AIRCRAFT: The detection is an aircraft.
        EQUIPMENT: The detection is a "connected" ground-service equipment class.
    """

    AIRCRAFT = "aircraft"
    EQUIPMENT = "equipment"


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in pixel coordinates (x2 >= x1, y2 >= y1 expected).

    Attributes:
        x1: Left edge.
        y1: Top edge.
        x2: Right edge.
        y2: Bottom edge.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Box width, never negative."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Box height, never negative."""
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Box area in px^2."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Geometric center (cx, cy)."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def intersection_area(self, other: "BoundingBox") -> float:
        """Area of the overlap between two boxes.

        Args:
            other: Box to intersect with.
        Returns:
            Overlap area in px^2, 0.0 when the boxes are disjoint.
        """
        ix = min(self.x2, other.x2) - max(self.x1, other.x1)
        iy = min(self.y2, other.y2) - max(self.y1, other.y1)
        if ix <= 0.0 or iy <= 0.0:
            return 0.0
        return ix * iy

    def iou(self, other: "BoundingBox") -> float:
        """Intersection-over-union of two boxes.

        Args:
            other: Box to compare with.
        Returns:
            IoU in [0, 1]; 0.0 when the union has zero area.
        """
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        return inter / union if union > 0.0 else 0.0

    def expanded(self, margin_ratio: float) -> "BoundingBox":
        """Return a copy grown on every side by a fraction of its own size.

        Args:
            margin_ratio: Margin as a fraction of width (horizontal) and height (vertical).
        Returns:
            A new, larger BoundingBox (not clipped to the frame).
        """
        mx = self.width * margin_ratio
        my = self.height * margin_ratio
        return BoundingBox(self.x1 - mx, self.y1 - my, self.x2 + mx, self.y2 + my)

    def anchor(self, mode: str = "center") -> Tuple[float, float]:
        """Reference point used for ROI membership tests.

        Args:
            mode: ``"center"`` (box center) or ``"bottom_center"`` (middle of bottom edge).
        Returns:
            The (x, y) anchor point.
        Raises:
            ValueError: If ``mode`` is not one of ANCHOR_MODES.
        """
        if mode == "center":
            return self.center
        if mode == "bottom_center":
            return ((self.x1 + self.x2) / 2.0, self.y2)
        raise ValueError(f"Unknown anchor mode {mode!r}; expected one of {ANCHOR_MODES}")

    def as_int_tuple(self) -> Tuple[int, int, int, int]:
        """Rounded integer corners (x1, y1, x2, y2) for drawing."""
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))


@dataclass(frozen=True)
class RawDetection:
    """Detector/tracker output before class mapping (model vocabulary).

    Attributes:
        box: Bounding box in frame pixels.
        confidence: Detector confidence in [0, 1].
        class_id: Integer class index of the model.
        label: Class name as stored in the model.
        track_id: Tracker-assigned id, or None when the tracker produced no id.
    """

    box: BoundingBox
    confidence: float
    class_id: int
    label: str
    track_id: Optional[int] = None


@dataclass(frozen=True)
class Detection:
    """Detection after class mapping (pipeline vocabulary).

    Attributes:
        box: Bounding box in frame pixels.
        confidence: Detector confidence in [0, 1].
        class_id: Integer class index of the model.
        label: Class name as stored in the model.
        track_id: Tracker-assigned id, or None.
        kind: AIRCRAFT or EQUIPMENT.
        service: The ServiceType for EQUIPMENT detections, None for aircraft.
    """

    box: BoundingBox
    confidence: float
    class_id: int
    label: str
    track_id: Optional[int]
    kind: DetectionKind
    service: Optional[ServiceType] = None


@dataclass(frozen=True)
class AircraftSnapshot:
    """Stable-identity view of one confirmed aircraft for the current frame.

    Attributes:
        aircraft_id: Pipeline-stable id (survives tracker id changes), unique per run.
        box: Last observed box (equals the current box when ``observed`` is True).
        confidence: Confidence of the last observation.
        observed: True if detected in the current frame, False if coasting on its last box.
        first_seen_ts: Stream timestamp of the first detection of this aircraft.
        last_seen_ts: Stream timestamp of the most recent detection.
        raw_track_id: ByteTrack id of the last observation, or None.
    """

    aircraft_id: int
    box: BoundingBox
    confidence: float
    observed: bool
    first_seen_ts: float
    last_seen_ts: float
    raw_track_id: Optional[int] = None


@dataclass(frozen=True)
class FramePacket:
    """One frame leaving the ingestion stage.

    Attributes:
        frame: BGR uint8 image of shape (H, W, 3).
        index: Monotonic index of emitted frames of this source (starts at 1).
        timestamp: Stream clock in epoch seconds (wall clock for live, video timeline for files).
        source_id: Human-readable identifier of the source.
        media_time_s: Seconds since the start of the video file, None for live streams.
    """

    frame: np.ndarray
    index: int
    timestamp: float
    source_id: str
    media_time_s: Optional[float] = None

    @property
    def size(self) -> Tuple[int, int]:
        """Frame size as (width, height)."""
        return (int(self.frame.shape[1]), int(self.frame.shape[0]))


@dataclass(frozen=True)
class RunInfo:
    """Metadata describing one pipeline run, handed to sinks at start.

    Attributes:
        run_id: Unique run identifier.
        started_ts: Wall-clock epoch seconds when the run started.
        source_uri: Source URL or path.
        config: JSON-serialisable snapshot of the effective configuration.
    """

    run_id: str
    started_ts: float
    source_uri: str
    config: Dict[str, Any] = field(default_factory=dict)

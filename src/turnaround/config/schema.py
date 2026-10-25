"""Typed configuration schema. Pure data: no I/O, no validation logic beyond defaults.

Every tunable of the pipeline lives here so that moving to another PC only
requires editing one YAML file. Validation lives in ``config.loader``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from turnaround.domain.types import SERVICE_NAMES


@dataclass
class SourceConfig:
    """Where frames come from.

    Attributes:
        kind: ``"youtube"`` (live/VOD page URL resolved by yt-dlp), ``"stream"`` (direct HLS/RTSP/HTTP URL) or ``"file"`` (recorded video replay).
        uri: YouTube page URL, stream URL or video file path (relative paths resolve against the project root for kind=file).
        target_fps: Maximum frames per second handed to the pipeline (extra frames are skipped).
        max_height: Preferred maximum video height when resolving YouTube streams.
        buffer_size: Bounded frame-buffer length for live sources (older frames are dropped when full).
        read_timeout_s: Network read timeout hint passed to the FFmpeg backend.
        reconnect_initial_s: First reconnect delay after a stream failure.
        reconnect_max_s: Upper bound of the exponential reconnect delay.
        realtime_pacing: File replay only: sleep so frames are emitted at real-time speed.
        loop: File replay only: restart the video at EOF (timeline keeps increasing).
        start_time_iso: File replay only: ISO-8601 wall-clock origin of the video timeline (default: now).
    """

    kind: str = "file"
    uri: str = ""
    target_fps: float = 5.0
    max_height: int = 720
    buffer_size: int = 2
    read_timeout_s: float = 20.0
    reconnect_initial_s: float = 2.0
    reconnect_max_s: float = 60.0
    realtime_pacing: bool = False
    loop: bool = False
    start_time_iso: Optional[str] = None


@dataclass
class ModelConfig:
    """Detector + tracker settings.

    Attributes:
        weights_path: Path to the YOLOv8 ``.pt`` weights (relative to the project root).
        device: ``"auto"`` (CUDA if available else CPU), ``"cpu"``, ``"cuda:0"`` or ``"0"``.
        half: FP16 inference: True/False, or None for automatic (on for CUDA only).
        imgsz: Inference image size.
        conf: Detector confidence floor (stage-specific thresholds are applied later).
        iou: NMS IoU threshold.
        tracker_config: Path to the ByteTrack YAML (relative to the project root).
        max_det: Maximum detections per frame.
    """

    weights_path: str = "weights/best.pt"
    device: str = "auto"
    half: Optional[bool] = None
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    tracker_config: str = "configs/bytetrack.yaml"
    max_det: int = 100


@dataclass
class ClassMapConfig:
    """Mapping from pipeline roles to model class names (or integer class ids).

    Names are compared after normalisation (lowercase, spaces/hyphens -> underscores).

    Attributes:
        aircraft: Model class for the aircraft.
        aerobridge: Model class meaning "aerobridge connected".
        catering: Model class meaning "catering truck connected".
        cargo: Model class meaning "cargo loader connected".
        pushback: Model class meaning "pushback tug connected".
    """

    aircraft: Union[str, int] = "aircraft"
    aerobridge: Union[str, int] = "aerobridge_connected"
    catering: Union[str, int] = "catering_truck_connected"
    cargo: Union[str, int] = "cargo_loader_connected"
    pushback: Union[str, int] = "pushback_tug_connected"


@dataclass
class RoiConfig:
    """Manually defined regions of interest.

    Attributes:
        enabled: Master switch; when False no ROI filtering happens.
        coordinate_mode: ``"normalized"`` (x,y in [0,1]) or ``"pixel"`` (needs reference_resolution).
        reference_resolution: [width, height] the pixel polygons were drawn at.
        anchor: Box point tested against polygons: ``"center"`` or ``"bottom_center"``.
        apply_to_aircraft: Also require aircraft detections to lie inside analysis_zone.
        analysis_zone: Polygon (list of [x, y]) outside of which detections are ignored.
        service_zones: Optional per-service polygons keyed by service name.
    """

    enabled: bool = False
    coordinate_mode: str = "normalized"
    reference_resolution: Optional[List[int]] = None
    anchor: str = "center"
    apply_to_aircraft: bool = True
    analysis_zone: List[List[float]] = field(default_factory=list)
    service_zones: Dict[str, List[List[float]]] = field(default_factory=dict)


@dataclass
class SceneGuardConfig:
    """Camera-view-change detection (protects the fixed manual ROI).

    Attributes:
        enabled: Master switch.
        reference_image: Optional image path defining the expected view (default: first frame).
        thumbnail_size: [width, height] of the edge-map signature.
        change_threshold: Similarity below this counts as a different view.
        recover_threshold: Similarity at/above this counts as the original view again.
        trigger_frames: Consecutive low-similarity frames required to declare a change.
        recover_frames: Consecutive high-similarity frames required to declare recovery.
        reference_blend: Per-frame exponential blend of the reference toward the current view while stable.
    """

    enabled: bool = True
    reference_image: Optional[str] = None
    thumbnail_size: List[int] = field(default_factory=lambda: [96, 54])
    change_threshold: float = 0.30
    recover_threshold: float = 0.55
    trigger_frames: int = 3
    recover_frames: int = 3
    reference_blend: float = 0.02


@dataclass
class AircraftConfig:
    """Aircraft identity resolution (stable ids across tracker id switches and occlusion).

    Attributes:
        min_confidence: Minimum detector confidence for an aircraft detection to be used.
        confirm_hits: Detections needed before a tentative aircraft is confirmed (opens a session).
        tentative_timeout_s: Observed seconds a tentative aircraft may stay unseen before being dropped.
        lost_timeout_s: Observed seconds a confirmed aircraft may stay unseen before it is declared departed.
        reid_iou_threshold: Minimum IoU to re-attach a new tracker id to a recently lost aircraft.
        duplicate_iou_threshold: Aircraft boxes overlapping above this IoU in one frame are merged (higher confidence wins).
    """

    min_confidence: float = 0.40
    confirm_hits: int = 3
    tentative_timeout_s: float = 3.0
    lost_timeout_s: float = 120.0
    reid_iou_threshold: float = 0.30
    duplicate_iou_threshold: float = 0.80


@dataclass
class AssociationConfig:
    """Equipment-to-aircraft association and per-service confidence floors.

    Attributes:
        aircraft_box_margin: Fraction by which aircraft boxes are enlarged before overlap testing.
        min_equipment_overlap: Required fraction of the equipment box lying inside the enlarged aircraft box.
        min_confidence: Per-service minimum confidence (keys: aerobridge, catering, cargo, pushback).
    """

    aircraft_box_margin: float = 0.15
    min_equipment_overlap: float = 0.20
    min_confidence: Dict[str, float] = field(default_factory=lambda: {name: 0.35 for name in SERVICE_NAMES})


@dataclass
class ServiceRuleConfig:
    """Hysteresis rule for one service.

    Attributes:
        on_window_s: Sliding window in which positive samples are counted for activation.
        on_min_hits: Positive samples required inside on_window_s to activate.
        off_grace_s: Observed seconds without positive evidence tolerated before release (occlusion tolerance).
        min_interval_s: Intervals shorter than this are retracted as false triggers on normal release.
    """

    on_window_s: float = 3.0
    on_min_hits: int = 4
    off_grace_s: float = 8.0
    min_interval_s: float = 5.0


@dataclass
class TemporalConfig:
    """Temporal reasoning parameters.

    Attributes:
        max_gap_s: Cap on the time delta credited between two processed frames (stalls/pauses are not counted as absence).
        partial_start_window_s: Aircraft first seen within this many seconds of the first processed frame are flagged partial_start.
        defaults: Rule applied to every service unless overridden.
        overrides: Per-service partial overrides, e.g. ``{"pushback": {"off_grace_s": 5.0}}``.
    """

    max_gap_s: float = 3.0
    partial_start_window_s: float = 5.0
    defaults: ServiceRuleConfig = field(default_factory=ServiceRuleConfig)
    overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def rule_for(self, service: str) -> ServiceRuleConfig:
        """Effective rule for a service (defaults merged with its override).

        Args:
            service: Service name (a ServiceType value).
        Returns:
            A new ServiceRuleConfig with overrides applied.
        """
        return dataclasses.replace(self.defaults, **self.overrides.get(service, {}))


@dataclass
class StorageConfig:
    """SQLite persistence.

    Attributes:
        db_path: SQLite file path (relative to the project root); parent directories are created.
    """

    db_path: str = "data/turnaround.db"


@dataclass
class OutputConfig:
    """Live view, recording and overlay options.

    Attributes:
        runtime_dir: Directory for latest.jpg and status.json (the dashboard's live feed).
        publish_interval_s: Minimum seconds between live-state file updates.
        jpeg_quality: JPEG quality of latest.jpg (1-100).
        draw_rejected: Draw rejected equipment detections in gray with the rejection reason.
        record_path: Optional path of an annotated output video (mp4/avi); None disables recording.
    """

    runtime_dir: str = "runtime"
    publish_interval_s: float = 0.5
    jpeg_quality: int = 80
    draw_rejected: bool = False
    record_path: Optional[str] = None


@dataclass
class LoggingConfig:
    """Logging options.

    Attributes:
        level: Python logging level name.
        file: Optional log file path (rotating); None logs to console only.
    """

    level: str = "INFO"
    file: Optional[str] = None


@dataclass
class AppConfig:
    """Root configuration object.

    Attributes:
        source: Frame source settings.
        model: Detector/tracker settings.
        classes: Model-class mapping.
        roi: Manual regions of interest.
        scene_guard: View-change guard.
        aircraft: Aircraft identity resolution.
        association: Equipment association.
        temporal: Hysteresis / temporal reasoning.
        storage: SQLite settings.
        output: Live view / recording settings.
        logging: Logging settings.
        config_path: Absolute path of the YAML file the config was loaded from (set by the loader).
    """

    source: SourceConfig = field(default_factory=SourceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    classes: ClassMapConfig = field(default_factory=ClassMapConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    scene_guard: SceneGuardConfig = field(default_factory=SceneGuardConfig)
    aircraft: AircraftConfig = field(default_factory=AircraftConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    config_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Plain nested-dict snapshot of the configuration (JSON-serialisable)."""
        return dataclasses.asdict(self)

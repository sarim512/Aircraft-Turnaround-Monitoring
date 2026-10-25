"""Composition root: the only module that knows every concrete class and wires them together."""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import List, Optional

from turnaround.config.schema import AppConfig
from turnaround.detection.base import ObjectTracker
from turnaround.detection.class_mapper import ClassMapper
from turnaround.detection.ultralytics_tracker import UltralyticsTracker
from turnaround.domain.types import RunInfo
from turnaround.ingestion.base import FrameSource
from turnaround.ingestion.factory import build_source
from turnaround.pipeline.orchestrator import PipelineRunner
from turnaround.pipeline.ports import EventSink, FrameSink
from turnaround.runtime.live_state import LiveStateWriter
from turnaround.spatial.association import EquipmentAssociator
from turnaround.spatial.roi import RoiSet
from turnaround.spatial.scene_guard import SceneGuard
from turnaround.storage.writer import SqliteEventWriter
from turnaround.temporal.aircraft_registry import AircraftRegistry
from turnaround.temporal.engine import TurnaroundEngine
from turnaround.visualization.annotator import FrameAnnotator
from turnaround.visualization.video_recorder import AnnotatedVideoWriter


def new_run_id() -> str:
    """Unique, sortable run id such as ``20260101-120000-a1b2c3``."""
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def build_pipeline(
    cfg: AppConfig,
    tracker: Optional[ObjectTracker] = None,
    source: Optional[FrameSource] = None,
) -> PipelineRunner:
    """Construct a ready-to-run PipelineRunner from a validated configuration.

    Args:
        cfg: Validated, path-resolved AppConfig.
        tracker: Optional detector/tracker replacement (tests inject a fake); None builds the
            Ultralytics YOLOv8 + ByteTrack adapter from ``cfg.model``.
        source: Optional frame source replacement; None builds one from ``cfg.source``.
    Returns:
        PipelineRunner (not yet running).
    Raises:
        FileNotFoundError: Model weights missing.
        ClassMappingError: Configured classes do not match the model's class names.
    Logic:
        1. Build and ``load`` the tracker; build the ClassMapper and validate it against the model names
           (fail fast before any frame is read).
        2. Build RoiSet, EquipmentAssociator, AircraftRegistry and TurnaroundEngine.
        3. Build the optional SceneGuard, the SQLite event sink, the live-state frame sink and the optional
           video recorder, plus the FrameAnnotator.
        4. Assemble RunInfo (id, start time, source, config snapshot) and the PipelineRunner.
    Side effects:
        Loads model weights (GPU memory), creates/opens the SQLite database and the runtime directory.
    """
    if tracker is None:
        tracker = UltralyticsTracker(
            weights_path=cfg.model.weights_path,
            device=cfg.model.device,
            half=cfg.model.half,
            imgsz=cfg.model.imgsz,
            conf=cfg.model.conf,
            iou=cfg.model.iou,
            tracker_config=cfg.model.tracker_config,
            max_det=cfg.model.max_det,
        )
    tracker.load()
    mapper = ClassMapper(cfg.classes)
    mapper.validate(tracker.class_names)

    roi = RoiSet(cfg.roi)
    engine = TurnaroundEngine(cfg.temporal, AircraftRegistry(cfg.aircraft), EquipmentAssociator(cfg.association, roi), roi)
    guard = SceneGuard(cfg.scene_guard) if cfg.scene_guard.enabled else None

    event_sinks: List[EventSink] = [SqliteEventWriter(cfg.storage.db_path)]
    frame_sinks: List[FrameSink] = [LiveStateWriter(cfg.output.runtime_dir, cfg.output.jpeg_quality, cfg.output.publish_interval_s)]
    if cfg.output.record_path:
        frame_sinks.append(AnnotatedVideoWriter(cfg.output.record_path, cfg.source.target_fps))

    run_info = RunInfo(run_id=new_run_id(), started_ts=time.time(), source_uri=cfg.source.uri, config=cfg.to_dict())
    return PipelineRunner(
        run_info=run_info,
        source=source if source is not None else build_source(cfg.source),
        tracker=tracker,
        class_mapper=mapper,
        engine=engine,
        event_sinks=event_sinks,
        frame_sinks=frame_sinks,
        scene_guard=guard,
        annotator=FrameAnnotator(draw_rejected=cfg.output.draw_rejected),
        roi=roi,
    )

"""End-to-end: synthetic video -> fake tracker -> orchestrator -> engine -> SQLite -> analytics + live files."""
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import cv2
import numpy as np
import pytest

from turnaround.analytics.summary import compute_session_summary
from turnaround.config.schema import AppConfig
from turnaround.detection.base import ObjectTracker
from turnaround.domain.types import BoundingBox, RawDetection
from turnaround.pipeline.builder import build_pipeline
from turnaround.runtime.live_state import LiveStateReader
from turnaround.storage.reader import AnalyticsReader

from helpers import AIRCRAFT_BOX, EQUIP_BOX

NAMES = {0: "aircraft", 1: "aerobridge_connected", 2: "catering_truck_connected", 3: "cargo_loader_connected", 4: "pushback_tug_connected"}
START_ISO = "2026-01-01T00:00:00"
FPS_VIDEO = 10
SIZE = (320, 180)


def background(seed: int) -> np.ndarray:
    """Random but reproducible textured background image for a given seed."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (SIZE[1] // 8, SIZE[0] // 8, 3), dtype=np.uint8)
    return cv2.resize(small, SIZE, interpolation=cv2.INTER_NEAREST)


def make_video(path: Path, seconds: int, bg_for: Callable[[float], int] = lambda t: 1) -> str:
    """Write a synthetic MJPG video whose background seed depends on time."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS_VIDEO, SIZE)
    for i in range(seconds * FPS_VIDEO):
        writer.write(background(bg_for(i / FPS_VIDEO)))
    writer.release()
    return str(path)


class ScriptedTracker(ObjectTracker):
    """Fake YOLO: returns detections as a function of the processed-frame time (5 FPS stride)."""

    def __init__(self, script: Callable[[float], Sequence[RawDetection]]):
        """Store the detection script and counters."""
        self._script = script
        self.calls = 0
        self.resets = 0

    def load(self):
        """No-op."""
        pass

    def track(self, frame):
        """Return the scripted detections for the current processed frame (t = calls * 0.2 s)."""
        t = self.calls * 0.2
        self.calls += 1
        return list(self._script(t))

    def reset(self):
        """Count tracker resets."""
        self.resets += 1

    @property
    def class_names(self) -> Dict[int, str]:
        """The 5-class model vocabulary."""
        return NAMES


def raw(class_id: int, box: BoundingBox, track_id=None) -> RawDetection:
    """Build a RawDetection with confidence 0.9."""
    return RawDetection(box, 0.9, class_id, NAMES[class_id], track_id)


def turnaround_script(t: float) -> List[RawDetection]:
    """Scripted ground truth: aircraft 8-58 s, aerobridge 12-40 s (20% dropped), catering 15-35 s, pushback 48-56 s."""
    out: List[RawDetection] = []
    if 8.0 <= t < 58.0:
        out.append(raw(0, AIRCRAFT_BOX, track_id=1))
    if 12.0 <= t < 40.0 and int(round(t / 0.2)) % 5 != 0:  # aerobridge with 20% missed frames
        out.append(raw(1, EQUIP_BOX, track_id=2))
    if 15.0 <= t < 35.0:
        out.append(raw(2, BoundingBox(120, 160, 190, 250), track_id=3))
    if 48.0 <= t < 56.0:
        out.append(raw(4, BoundingBox(400, 150, 480, 250), track_id=4))
    return out


def make_config(tmp_path: Path, video: str, **overrides) -> AppConfig:
    """AppConfig for a file replay with temp DB/runtime dirs; ``section__name`` keyword overrides."""
    cfg = AppConfig()
    cfg.source.kind, cfg.source.uri, cfg.source.target_fps, cfg.source.start_time_iso = "file", video, 5.0, START_ISO
    cfg.storage.db_path = str(tmp_path / "t.db")
    cfg.output.runtime_dir = str(tmp_path / "runtime")
    cfg.aircraft.lost_timeout_s = 3.0
    for key, value in overrides.items():
        section, name = key.split("__")
        setattr(getattr(cfg, section), name, value)
    return cfg


def test_full_turnaround_run_produces_correct_sessions_intervals_and_kpis(tmp_path):
    video = make_video(tmp_path / "v.avi", seconds=70)
    cfg = make_config(tmp_path, video, output__record_path=str(tmp_path / "annotated.avi"))
    tracker = ScriptedTracker(turnaround_script)
    runner = build_pipeline(cfg, tracker=tracker)
    stats = runner.run()
    assert stats.stop_reason == "stream_end" and stats.frames_processed == 350 and stats.pipeline_errors == 0 and stats.sink_errors == 0

    reader = AnalyticsReader(cfg.storage.db_path)
    run_id = reader.latest_run_id()
    assert reader.list_runs()[0].ended_ts is not None
    sessions = reader.list_sessions(run_id)
    assert len(sessions) == 1
    s = sessions[0]
    assert s.status == "closed" and s.closure_reason == "aircraft_departed" and not s.partial_start and not s.partial_end

    base = __import__("datetime").datetime.fromisoformat(START_ISO).timestamp()
    ivs = {iv.service: iv for iv in reader.list_intervals(run_id)}
    assert set(ivs) == {"aerobridge", "catering", "pushback"}
    assert ivs["aerobridge"].start_ts == pytest.approx(base + 12.2, abs=0.45)  # first non-dropped frame
    assert ivs["aerobridge"].end_ts == pytest.approx(base + 39.8, abs=0.45)
    assert ivs["catering"].start_ts == pytest.approx(base + 15.0, abs=0.45)
    assert ivs["catering"].end_ts == pytest.approx(base + 34.8, abs=0.45)
    assert ivs["pushback"].start_ts == pytest.approx(base + 48.0, abs=0.45)

    summary = compute_session_summary(s, list(ivs.values()), now=base + 70)
    assert summary.is_complete
    assert summary.turnaround_time_s == pytest.approx(48.0 - 12.2, abs=0.8)  # first connection -> pushback start
    assert summary.services["aerobridge"].total_s == pytest.approx(27.6, abs=0.9)
    assert summary.max_concurrent == 2 and summary.parallel_s == pytest.approx(19.8, abs=1.0)

    types = [e.event_type for e in reader.list_events(run_id, limit=100)]
    assert types.count("session_started") == 1 and types.count("session_ended") == 1 and "service_discarded" not in types

    live = LiveStateReader(cfg.output.runtime_dir)
    status = live.read_status()
    assert status["state"] == "finished" and status["run_id"] == run_id
    assert live.read_frame_bytes()[:2] == b"\xff\xd8"  # JPEG
    assert (tmp_path / "annotated.avi").stat().st_size > 1000


def test_camera_cut_pauses_analysis_and_freezes_temporal_state(tmp_path):
    # camera shows another view during 20-30 s of video time (background seed switches)
    video = make_video(tmp_path / "v.avi", seconds=50, bg_for=lambda t: 2 if 20.0 <= t < 30.0 else 1)

    def script(t):  # detections only matter while the original view is visible
        dets = [raw(0, AIRCRAFT_BOX, track_id=1)]
        if 5.0 <= t:
            dets.append(raw(1, EQUIP_BOX, track_id=2))
        return dets

    tracker = ScriptedTracker(script)
    cfg = make_config(tmp_path, video)
    runner = build_pipeline(cfg, tracker=tracker)
    stats = runner.run()
    assert stats.frames_paused >= 40  # ~10 s at 5 FPS minus trigger/recovery frames
    assert tracker.resets >= 2  # reset on cut and on restore
    reader = AnalyticsReader(cfg.storage.db_path)
    run_id = reader.latest_run_id()
    kinds = [e.event_type for e in reader.list_events(run_id, limit=100)]
    assert kinds.count("scene_changed") == 1 and kinds.count("scene_restored") == 1
    # the aerobridge interval was not split by the pause: absence during the cut is not counted
    ab = [iv for iv in reader.list_intervals(run_id) if iv.service == "aerobridge"]
    assert len(ab) == 1


def test_sink_failure_does_not_stop_the_run(tmp_path):
    video = make_video(tmp_path / "v.avi", seconds=10)
    cfg = make_config(tmp_path, video)
    runner = build_pipeline(cfg, tracker=ScriptedTracker(lambda t: [raw(0, AIRCRAFT_BOX, track_id=1)]))

    class Boom:
        def begin_run(self, run): pass
        def handle(self, events): raise RuntimeError("disk full")
        def end_run(self, ts): pass
        def close(self): pass

    runner._event_sinks.append(Boom())
    stats = runner.run()
    assert stats.sink_errors >= 1 and stats.frames_processed == 50


def test_tracker_exception_is_isolated_per_frame(tmp_path):
    video = make_video(tmp_path / "v.avi", seconds=6)
    cfg = make_config(tmp_path, video)

    def flaky(t):
        if abs(t - 2.0) < 1e-6:
            raise ValueError("cuda hiccup")
        return []

    stats = build_pipeline(cfg, tracker=ScriptedTracker(flaky)).run()
    assert stats.pipeline_errors == 1 and stats.frames_processed == 29


def test_max_frames_and_stop_request_flush_open_sessions(tmp_path):
    video = make_video(tmp_path / "v.avi", seconds=60)
    cfg = make_config(tmp_path, video)
    runner = build_pipeline(cfg, tracker=ScriptedTracker(lambda t: [raw(0, AIRCRAFT_BOX, 1), raw(1, EQUIP_BOX, 2)]))
    stats = runner.run(max_frames=100)
    assert stats.frames_processed == 100 and stats.stop_reason == "shutdown"
    reader = AnalyticsReader(cfg.storage.db_path)
    s = reader.list_sessions(reader.latest_run_id())[0]
    assert s.status == "closed" and s.closure_reason == "shutdown" and s.partial_end and s.partial_start
    assert reader.list_intervals(reader.latest_run_id())[0].end_reason == "shutdown"


def test_missing_video_ends_the_run_cleanly_without_raising(tmp_path):
    cfg = make_config(tmp_path, str(tmp_path / "does_not_exist.avi"))
    stats = build_pipeline(cfg, tracker=ScriptedTracker(lambda t: [])).run()
    assert stats.stop_reason == "source_error" and stats.frames_processed == 0
    reader = AnalyticsReader(cfg.storage.db_path)
    assert reader.list_runs()[0].ended_ts is not None  # run row closed, sinks released
    assert LiveStateReader(cfg.output.runtime_dir).read_status()["state"] == "finished"


def test_class_mapping_mismatch_fails_fast(tmp_path):
    video = make_video(tmp_path / "v.avi", seconds=2)
    cfg = make_config(tmp_path, video)
    cfg.classes.aircraft = "plane"
    from turnaround.detection.class_mapper import ClassMappingError

    with pytest.raises(ClassMappingError):
        build_pipeline(cfg, tracker=ScriptedTracker(lambda t: []))

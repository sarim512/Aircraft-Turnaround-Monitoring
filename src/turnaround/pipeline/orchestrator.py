"""PipelineRunner: the frame loop that wires source, tracker, engine, guards and sinks together."""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from turnaround.detection.base import ObjectTracker
from turnaround.detection.class_mapper import ClassMapper
from turnaround.domain.events import DomainEvent, EndReason, EventType
from turnaround.domain.types import FramePacket, RunInfo
from turnaround.ingestion.base import FrameSource
from turnaround.pipeline.ports import EventSink, FrameSink
from turnaround.spatial.roi import RoiSet
from turnaround.spatial.scene_guard import SceneGuard, SceneState
from turnaround.temporal.engine import FrameResult, TurnaroundEngine
from turnaround.visualization.annotator import FrameAnnotator

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_ERRORS = 50


@dataclass
class RunStats:
    """Counters returned by ``PipelineRunner.run``.

    Attributes:
        frames_processed: Frames that went through tracker and engine.
        frames_paused: Frames skipped because the scene guard reported a view change.
        events_emitted: Domain events produced (including the final flush).
        sink_errors: Exceptions raised by event sinks (logged, not fatal).
        pipeline_errors: Exceptions raised while processing a frame (logged, frame skipped).
        stop_reason: Why the loop ended (stream_end, shutdown, too_many_errors, source_error).
    """

    frames_processed: int = 0
    frames_paused: int = 0
    events_emitted: int = 0
    sink_errors: int = 0
    pipeline_errors: int = 0
    stop_reason: str = ""


class PipelineRunner:
    """Runs the per-frame pipeline: guard -> detect/track -> map -> engine -> sinks."""

    def __init__(
        self,
        run_info: RunInfo,
        source: FrameSource,
        tracker: ObjectTracker,
        class_mapper: ClassMapper,
        engine: TurnaroundEngine,
        event_sinks: Sequence[EventSink],
        frame_sinks: Sequence[FrameSink] = (),
        scene_guard: Optional[SceneGuard] = None,
        annotator: Optional[FrameAnnotator] = None,
        roi: Optional[RoiSet] = None,
        read_timeout_s: float = 1.0,
        idle_status_interval_s: float = 5.0,
    ) -> None:
        """Receive all collaborators (dependency injection; nothing is constructed here).

        Args:
            run_info: Run metadata handed to event sinks.
            source: Frame source.
            tracker: Loaded detector+tracker.
            class_mapper: Validated class mapper.
            engine: Temporal engine.
            event_sinks: Event consumers (e.g. SqliteEventWriter).
            frame_sinks: Annotated-frame consumers (live view, recorder).
            scene_guard: Optional view-change guard.
            annotator: Optional overlay renderer (needed for frame sinks to receive overlays).
            roi: ROI set for drawing.
            read_timeout_s: Max seconds to wait per source read (bounds stop latency).
            idle_status_interval_s: Seconds between status-only publishes while no frames arrive.
        Logic:
            Assign every collaborator to a private attribute (nothing is constructed here). Initialise runtime
            state: stop flag False, scene-paused flag False, idle-publish clock 0, consecutive-error counter 0, a
            rolling deque (maxlen 30) of (monotonic time, latency ms) samples, last timestamp None and an empty
            RunStats.
        """
        self._run = run_info
        self._source = source
        self._tracker = tracker
        self._mapper = class_mapper
        self._engine = engine
        self._event_sinks = list(event_sinks)
        self._frame_sinks = list(frame_sinks)
        self._guard = scene_guard
        self._annotator = annotator
        self._roi = roi
        self._read_timeout_s = read_timeout_s
        self._idle_interval = idle_status_interval_s
        self._stop_requested = False
        self._paused = False
        self._last_idle_publish = 0.0
        self._consecutive_errors = 0
        self._timings: Deque[Tuple[float, float]] = deque(maxlen=30)
        self._last_ts: Optional[float] = None
        self.stats = RunStats()

    def request_stop(self) -> None:
        """Ask the loop to stop after the current iteration (signal-handler safe)."""
        self._stop_requested = True

    def run(self, max_frames: Optional[int] = None) -> RunStats:
        """Run until the source is exhausted, a stop is requested, or ``max_frames`` is reached.

        Args:
            max_frames: Optional cap on processed frames (testing / demos).
        Returns:
            RunStats.
        Logic:
            1. Begin the run on all event sinks and start the source; a source that fails to start
               (missing file, bad URL setup) is logged, stop_reason becomes ``source_error`` and the run is
               finalized cleanly (run row ended, sinks closed) without raising.
            2. Loop: read a packet; None -> stop if the source is exhausted (STREAM_END) else publish an
               idle status and continue; otherwise process it, isolating exceptions per frame and aborting
               after MAX_CONSECUTIVE_ERRORS consecutive failures.
            3. KeyboardInterrupt behaves like a stop request.
            4. ``_finalize`` always runs: flush open sessions, publish the final status, close everything.
        Side effects:
            Starts/stops the source; writes to sinks; logs.
        """
        reason = EndReason.SHUTDOWN
        self.stats.stop_reason = "shutdown"
        for sink in self._event_sinks:
            sink.begin_run(self._run)
        try:
            self._source.start()
        except Exception:
            logger.exception("Could not start the frame source")
            self.stats.stop_reason = "source_error"
            self._finalize(EndReason.SHUTDOWN)
            return self.stats
        try:
            while not self._stop_requested:
                packet = self._source.read(self._read_timeout_s)
                if packet is None:
                    if self._source.exhausted:
                        reason = EndReason.STREAM_END
                        self.stats.stop_reason = "stream_end"
                        break
                    self._publish_idle()
                    continue
                try:
                    self._process_packet(packet)
                    self._consecutive_errors = 0
                except Exception:
                    self.stats.pipeline_errors += 1
                    self._consecutive_errors += 1
                    logger.exception("Processing of frame %d failed", packet.index)
                    if self._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        self.stats.stop_reason = "too_many_errors"
                        break
                if max_frames is not None and self.stats.frames_processed >= max_frames:
                    break
        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self._finalize(reason)
        return self.stats

    def _process_packet(self, packet: FramePacket) -> None:
        """Handle one frame.

        Args:
            packet: Frame from the source.
        Logic:
            1. Scene guard: CHANGED -> ``_on_scene_changed`` and return (no inference).
               Returning to STABLE after a pause -> ``_on_scene_restored``.
            2. ``tracker.track`` -> ``class_mapper.resolve_all`` -> ``engine.process``.
            3. Emit the events, record latency, publish the annotated frame + status.
        """
        self._last_ts = packet.timestamp
        started = time.perf_counter()
        if self._guard is not None:
            state = self._guard.evaluate(packet.frame)
            if state is SceneState.CHANGED:
                self._on_scene_changed(packet)
                return
            if self._paused:
                self._on_scene_restored(packet)
        raw = self._tracker.track(packet.frame)
        detections = self._mapper.resolve_all(raw)
        result = self._engine.process(packet.timestamp, detections, packet.size)
        self._emit(result.events)
        self.stats.frames_processed += 1
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._timings.append((time.monotonic(), latency_ms))
        self._publish(packet, result, banner=None, state="running", force=False)

    def _on_scene_changed(self, packet: FramePacket) -> None:
        """React to a camera view change.

        Args:
            packet: The offending frame.
        Logic:
            On the first changed frame: set the paused flag, reset the tracker, emit SCENE_CHANGED and
            publish forcibly. Every changed frame increments ``frames_paused`` and refreshes the live view
            with a warning banner (no inference, no engine update, so temporal state is frozen).
        """
        first = not self._paused
        if first:
            self._paused = True
            self._tracker.reset()
            score = self._guard.last_score if self._guard else None
            self._emit([DomainEvent(EventType.SCENE_CHANGED, packet.timestamp, payload={"similarity": score})])
        self.stats.frames_paused += 1
        self._publish(packet, None, banner="VIEW CHANGED - ANALYSIS PAUSED", state="paused_scene_change", force=first)

    def _on_scene_restored(self, packet: FramePacket) -> None:
        """Resume analysis after the original view returned.

        Args:
            packet: First frame of the restored view.
        Side effects:
            Clears the paused flag, resets the tracker, emits SCENE_RESTORED.
        """
        self._paused = False
        self._tracker.reset()
        self._emit([DomainEvent(EventType.SCENE_RESTORED, packet.timestamp)])

    def _emit(self, events: Sequence[DomainEvent]) -> None:
        """Log and deliver events to every event sink, isolating sink failures.

        Args:
            events: Events to deliver (empty is a no-op).
        Side effects:
            Calls ``handle`` on each sink; exceptions are logged and counted in ``sink_errors``.
        """
        if not events:
            return
        self.stats.events_emitted += len(events)
        for event in events:
            logger.info(
                "EVENT %-17s session=%s aircraft=%s service=%s ts=%.1f",
                event.event_type.value, event.session_no, event.aircraft_id,
                event.service.value if event.service else "-", event.ts,
            )
        for sink in self._event_sinks:
            try:
                sink.handle(events)
            except Exception:
                self.stats.sink_errors += 1
                logger.exception("Event sink %s failed", type(sink).__name__)

    def _fps_and_latency(self) -> Tuple[float, float]:
        """Rolling throughput and latency.

        Returns:
            (frames per second over the last <=30 frames, mean latency in ms); zeros when unknown.
        """
        if not self._timings:
            return 0.0, 0.0
        mean_latency = sum(lat for _, lat in self._timings) / len(self._timings)
        if len(self._timings) < 2:
            return 0.0, mean_latency
        span = self._timings[-1][0] - self._timings[0][0]
        fps = (len(self._timings) - 1) / span if span > 0 else 0.0
        return fps, mean_latency

    def _build_status(self, state: str, packet: Optional[FramePacket], result: Optional[FrameResult]) -> Dict[str, Any]:
        """Assemble the JSON status published to the dashboard.

        Args:
            state: running | paused_scene_change | waiting_for_frames | finished.
            packet: Latest frame packet, if any.
            result: Latest engine result, if any.
        Returns:
            Status dictionary (run id, state, fps, latency, counters, source stats, open sessions).
        Logic:
            Take the rolling fps/latency; build one dict per open engine session (session_no, aircraft_id, active
            service names); return run id, state, wall-clock update time, stream timestamp (packet's, else the last
            seen), frame index, rounded fps/latency, counters, source URI, source stats, last scene similarity (or
            None) and the session list.
        """
        fps, latency = self._fps_and_latency()
        sessions: List[Dict[str, Any]] = []
        for session in self._engine.open_sessions():
            sessions.append(
                {
                    "session_no": session.session_no,
                    "aircraft_id": session.aircraft_id,
                    "active_services": [s.value for s in session.active_services()],
                }
            )
        return {
            "run_id": self._run.run_id,
            "state": state,
            "updated_wall_ts": time.time(),
            "stream_ts": packet.timestamp if packet else self._last_ts,
            "frame_index": packet.index if packet else None,
            "fps": round(fps, 2),
            "latency_ms": round(latency, 1),
            "frames_processed": self.stats.frames_processed,
            "frames_paused": self.stats.frames_paused,
            "source": self._run.source_uri,
            "source_stats": self._source.stats(),
            "scene_similarity": self._guard.last_score if self._guard else None,
            "sessions": sessions,
        }

    def _publish(self, packet: FramePacket, result: Optional[FrameResult], banner: Optional[str], state: str, force: bool) -> None:
        """Render and deliver the frame + status to every frame sink.

        Args:
            packet: Current frame.
            result: Engine result (None while paused).
            banner: Optional warning banner text.
            state: Pipeline state label.
            force: Bypass sink throttling.
        Logic:
            No-op without frame sinks. Otherwise annotate (or pass through the raw frame when no
            annotator is configured) and call ``submit`` on each sink; sink exceptions are logged.
        """
        if not self._frame_sinks:
            return
        fps, latency = self._fps_and_latency()
        if self._annotator is not None:
            hud = [f"{fps:.1f} FPS | {latency:.0f} ms | frame {packet.index}"]
            frame = self._annotator.render(packet.frame, result, self._roi, banner=banner, hud_lines=hud)
        else:
            frame = packet.frame
        status = self._build_status(state, packet, result)
        for sink in self._frame_sinks:
            try:
                sink.submit(frame, status, force=force)
            except Exception:
                logger.exception("Frame sink %s failed", type(sink).__name__)

    def _publish_idle(self) -> None:
        """Publish a status-only update while no frames arrive (rate limited by ``idle_status_interval_s``)."""
        now = time.monotonic()
        if now - self._last_idle_publish < self._idle_interval:
            return
        self._last_idle_publish = now
        status = self._build_status("waiting_for_frames", None, None)
        for sink in self._frame_sinks:
            try:
                sink.submit(None, status, force=True)
            except Exception:
                logger.exception("Frame sink %s failed", type(sink).__name__)

    def _finalize(self, reason: EndReason) -> None:
        """Flush and release everything (always called).

        Args:
            reason: EndReason recorded on sessions closed by the flush.
        Logic:
            Stop the source; flush open sessions at the last frame timestamp (partial_end); publish a final
            ``finished`` status; end the run on event sinks; close frame and event sinks.
        """
        self._source.stop()
        flush_ts = self._engine.last_ts if self._engine.last_ts is not None else time.time()
        self._emit(self._engine.flush(flush_ts, reason))
        final = self._build_status("finished", None, None)
        for sink in self._frame_sinks:
            try:
                sink.submit(None, final, force=True)
                sink.close()
            except Exception:
                logger.exception("Frame sink %s failed during shutdown", type(sink).__name__)
        for sink in self._event_sinks:
            try:
                sink.end_run(time.time())
                sink.close()
            except Exception:
                logger.exception("Event sink %s failed during shutdown", type(sink).__name__)
        logger.info("Run finished: %s", self.stats)

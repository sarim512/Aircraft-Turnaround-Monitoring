"""Live stream source: background capture thread with throttling, drop-oldest buffering and reconnects."""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

from turnaround.domain.types import FramePacket
from turnaround.ingestion.base import FrameSource
from turnaround.ingestion.frame_buffer import LatestFrameBuffer

logger = logging.getLogger(__name__)

CaptureFactory = Callable[[str], Any]


def default_capture_factory(read_timeout_s: float) -> CaptureFactory:
    """Build a factory creating OpenCV FFmpeg captures with a network read timeout.

    Args:
        read_timeout_s: Read/write timeout hint for the FFmpeg backend, in seconds.
    Returns:
        A callable ``url -> cv2.VideoCapture``.
    Side effects:
        Sets ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` (rw_timeout in microseconds) if not already set.
    """
    import cv2

    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rw_timeout;{int(read_timeout_s * 1_000_000)}")

    def factory(url: str) -> Any:
        """Open ``url`` with the FFmpeg backend.

        Args:
            url: Stream URL or file path.
        Returns:
            An (possibly unopened) ``cv2.VideoCapture``; callers check ``isOpened()``.
        """
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    return factory


class StreamFrameSource(FrameSource):
    """Resilient live frame source.

    A daemon thread grabs frames continuously (so the decoder never backs up), keeps only
    frames spaced by 1/target_fps, pushes them into a bounded drop-oldest buffer and
    reconnects with exponential backoff (re-resolving the URL) after any failure.
    """

    def __init__(
        self,
        resolver: Callable[[], str],
        target_fps: float,
        buffer_size: int = 2,
        reconnect_initial_s: float = 2.0,
        reconnect_max_s: float = 60.0,
        source_id: str = "stream",
        capture_factory: Optional[CaptureFactory] = None,
        read_timeout_s: float = 20.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Configure the source (does not connect yet).

        Args:
            resolver: Zero-argument callable returning the current stream URL (re-called on every reconnect).
            target_fps: Maximum emitted frame rate.
            buffer_size: Bounded buffer length.
            reconnect_initial_s: First backoff delay.
            reconnect_max_s: Backoff ceiling.
            source_id: Identifier copied to each FramePacket.
            capture_factory: ``url -> capture`` factory; None builds the default OpenCV factory (injectable for tests).
            read_timeout_s: FFmpeg network timeout hint used by the default factory.
            clock: Wall-clock function used for FramePacket timestamps.
        """
        self._resolver = resolver
        self._interval = 1.0 / target_fps
        self._buffer: LatestFrameBuffer[FramePacket] = LatestFrameBuffer(buffer_size)
        self._initial = reconnect_initial_s
        self._max = reconnect_max_s
        self._source_id = source_id
        self._capture_factory = capture_factory
        self._read_timeout_s = read_timeout_s
        self._clock = clock
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_index = 0
        self._reconnects = 0
        self._last_frame_wall = 0.0

    def start(self) -> None:
        """Start the capture thread (no-op if already running).

        Side effects:
            Creates the default capture factory if none was injected; spawns a daemon thread.
        """
        if self._thread and self._thread.is_alive():
            return
        if self._capture_factory is None:
            self._capture_factory = default_capture_factory(self._read_timeout_s)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="frame-grabber", daemon=True)
        self._thread.start()

    def read(self, timeout_s: float = 1.0) -> Optional[FramePacket]:
        """Pop the oldest buffered frame.

        Args:
            timeout_s: Maximum seconds to wait.
        Returns:
            FramePacket or None on timeout.
        """
        return self._buffer.get(timeout_s)

    def stop(self) -> None:
        """Signal the thread to stop, close the buffer and join the thread (max 5 s)."""
        self._stop.set()
        self._buffer.close()
        if self._thread:
            self._thread.join(timeout=5.0)

    @property
    def exhausted(self) -> bool:
        """Always False: a live stream is never finished, only reconnecting."""
        return False

    def stats(self) -> Dict[str, float]:
        """Counters: frames_emitted, frames_dropped, reconnects, last_frame_age_s (inf before first frame)."""
        age = (time.time() - self._last_frame_wall) if self._last_frame_wall else float("inf")
        return {
            "frames_emitted": float(self._frame_index),
            "frames_dropped": float(self._buffer.dropped),
            "reconnects": float(self._reconnects),
            "last_frame_age_s": age,
        }

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential reconnect delay.

        Args:
            attempt: 1-based consecutive failure count.
        Returns:
            ``min(reconnect_max_s, reconnect_initial_s * 2**(attempt-1))`` seconds.
        """
        return min(self._max, self._initial * (2 ** max(0, attempt - 1)))

    def _run(self) -> None:
        """Thread body: connect, pump frames, reconnect with backoff until stopped.

        Logic:
            1. ``_open`` (resolve URL + open capture); on failure back off and retry.
            2. ``_pump`` until the stream fails or stop is requested.
            3. Release the capture; if frames were received reset the failure counter.
            4. Count a reconnect, wait the backoff delay (interruptible by stop) and loop.
        """
        attempt = 0
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                attempt += 1
                self._stop.wait(self._backoff_delay(attempt))
                continue
            emitted = self._pump(cap)
            try:
                cap.release()
            except Exception:  # pragma: no cover - defensive
                logger.debug("capture release failed", exc_info=True)
            if self._stop.is_set():
                break
            attempt = 1 if emitted > 0 else attempt + 1
            self._reconnects += 1
            delay = self._backoff_delay(attempt)
            logger.warning("Stream interrupted after %d frames; reconnecting in %.1fs", emitted, delay)
            self._stop.wait(delay)

    def _open(self) -> Optional[Any]:
        """Resolve the URL and open a capture.

        Returns:
            An opened capture, or None if resolution or opening failed (logged).
        """
        try:
            url = self._resolver()
            cap = self._capture_factory(url)  # type: ignore[misc]
        except Exception as exc:
            logger.error("Could not resolve/open stream: %s", exc)
            return None
        if not cap.isOpened():
            logger.error("Capture failed to open")
            try:
                cap.release()
            except Exception:  # pragma: no cover - defensive
                pass
            return None
        logger.info("Stream opened")
        return cap

    def _pump(self, cap: Any) -> int:
        """Read frames from an open capture until it fails or stop is requested.

        Args:
            cap: Opened capture exposing grab() and retrieve().
        Returns:
            Number of frames pushed to the buffer during this connection.
        Logic:
            1. ``grab()`` every iteration so the decoder is drained at stream speed.
            2. Skip ``retrieve()`` unless at least 1/target_fps seconds passed since the last emitted frame.
            3. Wrap the frame into a FramePacket stamped with ``clock()`` and put it into the buffer.
            4. A failed grab ends the loop (caller reconnects).
        """
        emitted = 0
        last_emit = float("-inf")
        while not self._stop.is_set():
            if not cap.grab():
                break
            now = time.monotonic()
            if now - last_emit < self._interval:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            last_emit = now
            self._frame_index += 1
            self._last_frame_wall = time.time()
            self._buffer.put(FramePacket(frame=frame, index=self._frame_index, timestamp=self._clock(), source_id=self._source_id))
            emitted += 1
        return emitted

"""Recorded-video replay source (reproducible demos, development and tests)."""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional

from turnaround.domain.types import FramePacket
from turnaround.ingestion.base import FrameSource

logger = logging.getLogger(__name__)


class VideoFileSource(FrameSource):
    """Sequential video reader that subsamples to ``target_fps`` and stamps video-timeline timestamps.

    Timestamps are ``start_epoch + frame_no / native_fps`` so analytics are identical whether the
    replay is paced in real time or run as fast as possible.
    """

    def __init__(
        self,
        path: str,
        target_fps: float,
        realtime_pacing: bool = False,
        loop: bool = False,
        start_epoch: Optional[float] = None,
        source_id: Optional[str] = None,
    ) -> None:
        """Configure the reader (opens the file in ``start``).

        Args:
            path: Video file path.
            target_fps: Desired emitted frame rate (frames are skipped by an integer stride).
            realtime_pacing: Sleep so emission follows the video timeline in real time.
            loop: Restart at EOF; the timeline keeps increasing across loops.
            start_epoch: Epoch seconds assigned to video time 0 (None -> time of ``start``).
            source_id: Identifier copied to each FramePacket (default: the path).
        """
        self._path = path
        self._target_fps = target_fps
        self._pacing = realtime_pacing
        self._loop = loop
        self._start_epoch = start_epoch
        self._source_id = source_id or path
        self._cap = None
        self._native_fps = 25.0
        self._stride = 1
        self._base_epoch = 0.0
        self._frame_no = 0
        self._loop_offset_s = 0.0
        self._emitted = 0
        self._wall_start = 0.0
        self._exhausted = False

    def start(self) -> None:
        """Open the video and derive the frame stride.

        Raises:
            FileNotFoundError: If the file cannot be opened by OpenCV.
        Logic:
            Read native fps (fallback 25 when invalid); ``stride = max(1, round(native_fps / target_fps))``;
            fix the timeline origin and the pacing wall-clock origin.
        """
        import cv2

        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {self._path}")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._native_fps = fps if fps > 1e-3 else 25.0
        self._stride = max(1, int(round(self._native_fps / self._target_fps)))
        self._base_epoch = self._start_epoch if self._start_epoch is not None else time.time()
        self._wall_start = time.monotonic()
        self._frame_no = 0
        self._loop_offset_s = 0.0
        self._exhausted = False
        logger.info("Replaying %s (native %.2f fps, stride %d)", self._path, self._native_fps, self._stride)

    def read(self, timeout_s: float = 1.0) -> Optional[FramePacket]:
        """Return the next sampled frame.

        Args:
            timeout_s: Unused (reads are blocking and fast); kept for interface compatibility.
        Returns:
            FramePacket, or None at EOF (then ``exhausted`` is True).
        Logic:
            1. ``grab()`` each frame; on failure either rewind (loop: add the pass duration to the
               timeline offset) or mark exhausted and return None.
            2. Skip frames whose ``frame_no % stride != 0``.
            3. ``retrieve()``; compute media_time = frame_no/native_fps + loop offset.
            4. If pacing is enabled sleep until ``wall_start + media_time``.
        """
        import cv2

        if self._cap is None or self._exhausted:
            return None
        while True:
            if not self._cap.grab():
                if self._loop and self._frame_no > 0:
                    self._loop_offset_s += self._frame_no / self._native_fps
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._frame_no = 0
                    continue
                self._exhausted = True
                return None
            this_no = self._frame_no
            self._frame_no += 1
            if this_no % self._stride != 0:
                continue
            ok, frame = self._cap.retrieve()
            if not ok or frame is None:
                continue
            media_time = this_no / self._native_fps + self._loop_offset_s
            if self._pacing:
                delay = self._wall_start + media_time - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            self._emitted += 1
            return FramePacket(
                frame=frame,
                index=self._emitted,
                timestamp=self._base_epoch + media_time,
                source_id=self._source_id,
                media_time_s=media_time,
            )

    def stop(self) -> None:
        """Release the video capture (idempotent)."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def exhausted(self) -> bool:
        """True after EOF was reached and looping is disabled."""
        return self._exhausted

    def stats(self) -> Dict[str, float]:
        """Counters: frames_emitted, native_fps, stride."""
        return {"frames_emitted": float(self._emitted), "native_fps": self._native_fps, "stride": float(self._stride)}

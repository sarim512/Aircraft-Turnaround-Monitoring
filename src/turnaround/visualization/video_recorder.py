"""Optional annotated-video recorder (FrameSink)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AnnotatedVideoWriter:
    """Writes annotated frames to a video file; the writer opens lazily on the first frame."""

    def __init__(self, path: str, fps: float) -> None:
        """Remember the target.

        Args:
            path: Output file; ``.avi`` uses MJPG, anything else mp4v.
            fps: Frame rate stored in the file (use the source target_fps).
        """
        self._path = path
        self._fps = fps
        self._writer: Optional[cv2.VideoWriter] = None

    def submit(self, frame: Optional[np.ndarray], status: Mapping[str, Any], force: bool = False) -> None:
        """Append a frame.

        Args:
            frame: Annotated BGR frame; None is ignored.
            status: Ignored (present to satisfy the FrameSink protocol).
            force: Ignored.
        Side effects:
            Creates the parent directory and the video file on first use.
        """
        if frame is None:
            return
        if self._writer is None:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*("MJPG" if self._path.lower().endswith(".avi") else "mp4v"))
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(self._path, fourcc, self._fps, (w, h))
            if not self._writer.isOpened():
                logger.error("Could not open video writer for %s", self._path)
        if self._writer is not None and self._writer.isOpened():
            self._writer.write(frame)

    def close(self) -> None:
        """Finalize and release the video file (idempotent)."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

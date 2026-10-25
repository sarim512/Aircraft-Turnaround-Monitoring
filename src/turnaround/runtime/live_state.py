"""File-based live state (latest annotated frame + status JSON) shared with the dashboard.

The pipeline and the dashboard never import each other: they communicate through the SQLite
database and the two files written here (atomic replace, so readers never see partial files).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FRAME_FILENAME = "latest.jpg"
STATUS_FILENAME = "status.json"


def _atomic_write_bytes(path: Path, data: bytes) -> bool:
    """Write bytes atomically (tmp file + ``os.replace``).

    Args:
        path: Destination file.
        data: Content.
    Returns:
        True on success, False when the replace failed (e.g. Windows file locked by a reader).
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return True
    except OSError:
        logger.debug("atomic write to %s failed", path, exc_info=True)
        return False


class LiveStateWriter:
    """FrameSink writing ``latest.jpg`` and ``status.json`` at a throttled rate."""

    def __init__(self, runtime_dir: str, jpeg_quality: int = 80, min_interval_s: float = 0.5) -> None:
        """Prepare the output directory.

        Args:
            runtime_dir: Directory for the two files (created if missing).
            jpeg_quality: JPEG quality 1-100.
            min_interval_s: Minimum wall-clock seconds between two publishes.
        """
        self._dir = Path(runtime_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._quality = int(jpeg_quality)
        self._min_interval = float(min_interval_s)
        self._last_publish = 0.0

    def submit(self, frame: Optional[np.ndarray], status: Mapping[str, Any], force: bool = False) -> None:
        """Publish the status (and the frame, if given), respecting the throttle.

        Args:
            frame: Annotated BGR frame, or None to publish status only.
            status: JSON-serialisable status mapping.
            force: Bypass the throttle (used for state changes and the final status).
        Logic:
            Skip if less than ``min_interval_s`` elapsed and not forced. Encode the frame as JPEG and
            write it atomically, then write the status JSON atomically (status is written last so a
            reader that sees a new status also finds a frame at least as new).
        """
        now = time.monotonic()
        if not force and now - self._last_publish < self._min_interval:
            return
        self._last_publish = now
        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
            if ok:
                _atomic_write_bytes(self._dir / FRAME_FILENAME, buf.tobytes())
        _atomic_write_bytes(self._dir / STATUS_FILENAME, json.dumps(dict(status), default=str).encode("utf-8"))

    def close(self) -> None:
        """No resources to release (present to satisfy the FrameSink protocol)."""


class LiveStateReader:
    """Reads the files produced by LiveStateWriter (used by the dashboard)."""

    def __init__(self, runtime_dir: str) -> None:
        """Remember the directory.

        Args:
            runtime_dir: Directory containing latest.jpg and status.json.
        """
        self._dir = Path(runtime_dir)

    def read_status(self) -> Optional[Dict[str, Any]]:
        """Parsed status.json, or None if missing/unreadable (e.g. being replaced)."""
        try:
            return json.loads((self._dir / STATUS_FILENAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def read_frame_bytes(self) -> Optional[bytes]:
        """Raw JPEG bytes of latest.jpg, or None if missing/unreadable."""
        try:
            return (self._dir / FRAME_FILENAME).read_bytes()
        except OSError:
            return None

    def frame_age_s(self) -> Optional[float]:
        """Seconds since latest.jpg was modified, or None if it does not exist."""
        try:
            return max(0.0, time.time() - (self._dir / FRAME_FILENAME).stat().st_mtime)
        except OSError:
            return None

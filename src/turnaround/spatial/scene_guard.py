"""Detects camera view changes (PTZ moves, camera switches) that would invalidate the manual ROI."""
from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from turnaround.config.schema import SceneGuardConfig

logger = logging.getLogger(__name__)


class SceneState(str, Enum):
    """Result of the guard for a frame.

    Attributes:
        STABLE: The view matches the reference; analysis may proceed.
        CHANGED: The view differs from the reference; analysis must be paused.
    """

    STABLE = "stable"
    CHANGED = "changed"


class SceneGuard:
    """Compares a gradient-magnitude thumbnail of each frame with a reference signature.

    Edge maps make the comparison robust to day/night brightness shifts; the reference slowly
    follows the stable view (exponential blend) and is frozen while a change is active, so recovery
    is judged against the last good view.
    """

    def __init__(self, cfg: SceneGuardConfig) -> None:
        """Create the guard, optionally loading a reference image.

        Args:
            cfg: Scene guard configuration.
        Raises:
            FileNotFoundError: ``reference_image`` is set but cannot be read.
        """
        self._cfg = cfg
        self._size = (int(cfg.thumbnail_size[0]), int(cfg.thumbnail_size[1]))
        self._reference: Optional[np.ndarray] = None
        self._state = SceneState.STABLE
        self._bad = 0
        self._good = 0
        self._last_score = 1.0
        if cfg.reference_image:
            img = cv2.imread(cfg.reference_image)
            if img is None:
                raise FileNotFoundError(f"Cannot read scene_guard.reference_image: {cfg.reference_image}")
            self._reference = self._signature(img)

    @property
    def last_score(self) -> float:
        """Similarity score computed for the most recent frame (1.0 before any frame)."""
        return self._last_score

    @property
    def state(self) -> SceneState:
        """Current state."""
        return self._state

    def _signature(self, frame: np.ndarray) -> np.ndarray:
        """Compute the comparison signature of a frame.

        Args:
            frame: BGR (or gray) image.
        Returns:
            float32 gradient-magnitude map of the blurred, downsized grayscale image.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(gray, self._size, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (3, 3), 0).astype(np.float32)
        gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)

    @staticmethod
    def _similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Pearson correlation of two signatures.

        Args:
            a: First signature.
            b: Second signature (same shape).
        Returns:
            Correlation in [-1, 1]; 0.0 when either signature has no variance.
        """
        av = a.ravel() - a.mean()
        bv = b.ravel() - b.mean()
        denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
        if denom < 1e-9:
            return 0.0
        return float(np.dot(av, bv) / denom)

    def evaluate(self, frame: np.ndarray) -> SceneState:
        """Update the guard with a new frame.

        Args:
            frame: BGR image.
        Returns:
            Current SceneState after this frame.
        Logic:
            1. First frame without reference image becomes the reference (STABLE).
            2. STABLE: score >= change_threshold -> reset bad counter and blend the reference toward
               the frame; otherwise count a bad frame and switch to CHANGED after ``trigger_frames``
               consecutive bad frames (reference not updated while bad).
            3. CHANGED: score >= recover_threshold -> count good frames; after ``recover_frames``
               consecutive good frames return to STABLE; any lower score resets the good counter.
        """
        sig = self._signature(frame)
        if self._reference is None:
            self._reference = sig
            self._last_score = 1.0
            return self._state
        score = self._similarity(sig, self._reference)
        self._last_score = score
        cfg = self._cfg
        if self._state is SceneState.STABLE:
            if score >= cfg.change_threshold:
                self._bad = 0
                self._reference = (1.0 - cfg.reference_blend) * self._reference + cfg.reference_blend * sig
            else:
                self._bad += 1
                if self._bad >= cfg.trigger_frames:
                    self._state = SceneState.CHANGED
                    self._good = 0
                    logger.warning("Scene change detected (similarity %.2f)", score)
        else:
            if score >= cfg.recover_threshold:
                self._good += 1
                if self._good >= cfg.recover_frames:
                    self._state = SceneState.STABLE
                    self._bad = 0
                    logger.info("Scene restored (similarity %.2f)", score)
            else:
                self._good = 0
        return self._state

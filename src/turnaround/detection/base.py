"""Abstract detector+tracker contract so the model backend can be swapped without touching the pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from turnaround.domain.types import RawDetection


class ObjectTracker(ABC):
    """A stateful object detector that also assigns persistent track ids across frames."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights and warm the backend up (raises if weights are missing)."""

    @abstractmethod
    def track(self, frame: np.ndarray) -> List[RawDetection]:
        """Detect and track objects in one frame.

        Args:
            frame: BGR uint8 image (H, W, 3).
        Returns:
            Detections in model vocabulary with tracker ids (None when no id was assigned).
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear tracker state (call after a camera cut so stale tracks are not carried over)."""

    @property
    @abstractmethod
    def class_names(self) -> Dict[int, str]:
        """Mapping class id -> class name of the loaded model."""

    @property
    def device_description(self) -> str:
        """Human-readable description of the compute device (default: 'unknown')."""
        return "unknown"

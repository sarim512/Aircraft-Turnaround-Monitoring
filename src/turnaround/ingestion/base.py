"""Abstract frame-source contract implemented by every ingestion adapter."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

from turnaround.domain.types import FramePacket


class FrameSource(ABC):
    """Contract between the orchestrator and any frame provider (live or recorded)."""

    @abstractmethod
    def start(self) -> None:
        """Open the source and begin producing frames (may spawn a background thread)."""

    @abstractmethod
    def read(self, timeout_s: float = 1.0) -> Optional[FramePacket]:
        """Fetch the next frame.

        Args:
            timeout_s: Maximum seconds to wait for a frame.
        Returns:
            The next FramePacket, or None if none is available (timeout, or end of a finite source).
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop producing frames and release all resources (idempotent)."""

    @property
    @abstractmethod
    def exhausted(self) -> bool:
        """True once a finite source can never deliver another frame (always False for live sources)."""

    def stats(self) -> Dict[str, float]:
        """Diagnostic counters (frames emitted, dropped, reconnects, ...). Default: empty."""
        return {}

    def __enter__(self) -> "FrameSource":
        """Start the source when entering a ``with`` block and return it."""
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Stop the source when leaving a ``with`` block."""
        self.stop()

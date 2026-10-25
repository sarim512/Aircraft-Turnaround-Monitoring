"""Structural interfaces (ports) the orchestrator depends on; adapters implement them."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np

from turnaround.domain.events import DomainEvent
from turnaround.domain.types import RunInfo


class EventSink(Protocol):
    """Consumer of domain events (database, alerting, message bus, ...)."""

    def begin_run(self, run: RunInfo) -> None:
        """Called once before the first frame.

        Args:
            run: Metadata of the run.
        """
        ...

    def handle(self, events: Sequence[DomainEvent]) -> None:
        """Called with the events of one frame (non-empty).

        Args:
            events: Events in emission order.
        """
        ...

    def end_run(self, ended_ts: float) -> None:
        """Called once after the final flush.

        Args:
            ended_ts: Wall-clock end time.
        """
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class FrameSink(Protocol):
    """Consumer of annotated frames plus a status mapping (live view, recorder, streamer)."""

    def submit(self, frame: Optional[np.ndarray], status: Mapping[str, Any], force: bool = False) -> None:
        """Receive one frame and status.

        Args:
            frame: Annotated BGR frame, or None for status-only updates.
            status: JSON-serialisable status mapping.
            force: Bypass any throttling the sink applies.
        """
        ...

    def close(self) -> None:
        """Release resources."""
        ...

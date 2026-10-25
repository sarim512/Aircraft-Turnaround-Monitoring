"""Thread-safe bounded buffer that drops the oldest item when full."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Generic, Optional, TypeVar

T = TypeVar("T")


class LatestFrameBuffer(Generic[T]):
    """Bounded FIFO between a capture thread and the inference loop.

    Keeps latency bounded: when the consumer is slower than the producer the oldest
    items are discarded and counted, so inference never falls behind real time.
    """

    def __init__(self, maxsize: int = 2) -> None:
        """Create the buffer.

        Args:
            maxsize: Maximum number of buffered items (>= 1).
        Raises:
            ValueError: If maxsize < 1.
        """
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._items: Deque[T] = deque()
        self._maxsize = maxsize
        self._cond = threading.Condition()
        self._dropped = 0
        self._closed = False

    def put(self, item: T) -> bool:
        """Append an item, evicting the oldest one if the buffer is full.

        Args:
            item: Object to enqueue.
        Returns:
            True if an older item had to be dropped to make room.
        Side effects:
            Increments the dropped counter and wakes one waiting consumer.
        """
        with self._cond:
            dropped = False
            if len(self._items) >= self._maxsize:
                self._items.popleft()
                self._dropped += 1
                dropped = True
            self._items.append(item)
            self._cond.notify()
            return dropped

    def get(self, timeout_s: float = 1.0) -> Optional[T]:
        """Remove and return the oldest buffered item.

        Args:
            timeout_s: Maximum seconds to wait for an item.
        Returns:
            The item, or None on timeout or when the buffer was closed while empty.
        Logic:
            Loop on the condition variable until an item exists, the buffer is closed, or the
            deadline passes (guards against spurious wake-ups).
        """
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while not self._items:
                if self._closed:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._items.popleft()

    def close(self) -> None:
        """Mark the buffer closed and wake all waiting consumers."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def dropped(self) -> int:
        """Number of items evicted because the consumer was too slow."""
        with self._cond:
            return self._dropped

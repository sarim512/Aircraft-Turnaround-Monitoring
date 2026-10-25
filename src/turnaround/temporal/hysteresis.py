"""Debounced boolean state with n-of-m activation and grace-period release (flicker/occlusion safe)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional

from turnaround.config.schema import ServiceRuleConfig


class TransitionKind(str, Enum):
    """Direction of a state change.

    Attributes:
        ACTIVATED: Inactive -> active.
        DEACTIVATED: Active -> inactive.
    """

    ACTIVATED = "activated"
    DEACTIVATED = "deactivated"


@dataclass(frozen=True)
class Transition:
    """A state change reported by the filter.

    Attributes:
        kind: ACTIVATED or DEACTIVATED.
        effective_ts: When the change really happened (first positive for activation, last positive for release).
        decided_ts: Timestamp of the frame in which the filter decided (>= effective_ts).
        start_ts: Effective start of the active interval (equals effective_ts for ACTIVATED).
    """

    kind: TransitionKind
    effective_ts: float
    decided_ts: float
    start_ts: float


class HysteresisFilter:
    """Turns a noisy per-frame boolean into a stable interval signal.

    Activation: at least ``on_min_hits`` positive samples inside the last ``on_window_s`` seconds
    (negatives do not reset, so flicker is tolerated). Release: ``off_grace_s`` of *observed* time
    without any positive sample (occlusion/missed-detection tolerant). Interval boundaries are
    back-dated to real evidence, removing the detection-latency bias.
    """

    def __init__(self, rule: ServiceRuleConfig) -> None:
        """Create an inactive filter.

        Args:
            rule: Hysteresis parameters.
        """
        self._rule = rule
        self._positives: Deque[float] = deque()
        self._active = False
        self._active_since: Optional[float] = None
        self._last_positive_ts: Optional[float] = None
        self._absent_s = 0.0

    @property
    def active(self) -> bool:
        """True while the debounced state is active."""
        return self._active

    @property
    def active_since(self) -> Optional[float]:
        """Effective start timestamp of the current interval (None when inactive)."""
        return self._active_since

    @property
    def last_positive_ts(self) -> Optional[float]:
        """Timestamp of the latest positive sample ever seen (None before the first)."""
        return self._last_positive_ts

    def update(self, ts: float, dt: float, positive: bool) -> Optional[Transition]:
        """Feed one observation.

        Args:
            ts: Frame timestamp.
            dt: Observed seconds since the previous update (already gap-capped by the engine).
            positive: Whether valid evidence was present in this frame.
        Returns:
            A Transition if the debounced state changed in this call, else None.
        Logic:
            1. Positive: remember ``last_positive_ts``, reset the absence counter, and (if inactive)
               append ``ts`` to the positives deque. Negative: add ``dt`` to the absence counter.
            2. Inactive: drop positives older than ``ts - on_window_s``; if ``on_min_hits`` remain,
               activate with ``active_since`` = the oldest remaining positive (back-dated start).
            3. Active: if the absence counter >= ``off_grace_s`` deactivate with effective end =
               ``last_positive_ts`` (back-dated end).
        """
        rule = self._rule
        if positive:
            self._last_positive_ts = ts
            self._absent_s = 0.0
            if not self._active:
                self._positives.append(ts)
        else:
            self._absent_s += max(0.0, dt)

        if not self._active:
            cutoff = ts - rule.on_window_s
            while self._positives and self._positives[0] < cutoff:
                self._positives.popleft()
            if len(self._positives) >= rule.on_min_hits:
                self._active = True
                self._active_since = self._positives[0]
                self._positives.clear()
                return Transition(TransitionKind.ACTIVATED, self._active_since, ts, self._active_since)
            return None

        if self._absent_s >= rule.off_grace_s:
            return self._deactivate(ts)
        return None

    def force_close(self, ts: float) -> Optional[Transition]:
        """Close an active interval immediately (session end / shutdown) and clear pending evidence.

        Args:
            ts: Decision timestamp.
        Returns:
            DEACTIVATED Transition if the filter was active, else None.
        """
        if not self._active:
            self._positives.clear()
            return None
        return self._deactivate(ts)

    def _deactivate(self, decided_ts: float) -> Transition:
        """Leave the active state and build the closing Transition.

        Args:
            decided_ts: Timestamp at which the decision was made.
        Returns:
            DEACTIVATED Transition whose effective_ts is the last positive sample.
        """
        start = self._active_since if self._active_since is not None else decided_ts
        end = self._last_positive_ts if self._last_positive_ts is not None else start
        self._active = False
        self._active_since = None
        self._absent_s = 0.0
        self._positives.clear()
        return Transition(TransitionKind.DEACTIVATED, end, decided_ts, start)

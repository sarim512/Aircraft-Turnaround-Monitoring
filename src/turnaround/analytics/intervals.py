"""Pure interval arithmetic used by the turnaround analytics."""
from __future__ import annotations

from typing import List, Sequence, Tuple

Interval = Tuple[float, float]


def merge_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    """Union of intervals as a sorted list of disjoint intervals.

    Args:
        intervals: (start, end) pairs, possibly overlapping/unsorted; pairs with end <= start are dropped.
    Returns:
        Sorted disjoint intervals; touching intervals are merged.
    """
    valid = sorted((s, e) for s, e in intervals if e > s)
    merged: List[Interval] = []
    for start, end in valid:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def total_length(intervals: Sequence[Interval]) -> float:
    """Total covered length of already-disjoint intervals.

    Args:
        intervals: Disjoint intervals.
    Returns:
        Sum of (end - start).
    """
    return float(sum(e - s for s, e in intervals))


def clip_intervals(intervals: Sequence[Interval], lo: float, hi: float) -> List[Interval]:
    """Clip intervals to the window [lo, hi].

    Args:
        intervals: Intervals to clip.
        lo: Window start.
        hi: Window end.
    Returns:
        Intervals intersected with the window; empty results are dropped.
    """
    clipped = [(max(s, lo), min(e, hi)) for s, e in intervals]
    return [(s, e) for s, e in clipped if e > s]


def complement(window: Interval, disjoint: Sequence[Interval]) -> List[Interval]:
    """Gaps of a window not covered by disjoint intervals.

    Args:
        window: (start, end) window.
        disjoint: Sorted disjoint intervals already clipped to the window.
    Returns:
        Sorted gaps with positive length.
    """
    gaps: List[Interval] = []
    cursor = window[0]
    for s, e in disjoint:
        if s > cursor:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < window[1]:
        gaps.append((cursor, window[1]))
    return gaps


def concurrency_profile(interval_groups: Sequence[Sequence[Interval]]) -> Tuple[float, int]:
    """Time with at least two groups active and the maximum number of simultaneously active groups.

    Args:
        interval_groups: One list of disjoint intervals per group (e.g. per service).
    Returns:
        (seconds with >= 2 groups active, maximum simultaneous groups).
    Logic:
        Sweep line over +1/-1 boundary events sorted by time with ends before starts at equal times;
        accumulate the time spent while the counter is >= 2 and track its maximum.
    """
    boundaries: List[Tuple[float, int]] = []
    for group in interval_groups:
        for s, e in group:
            boundaries.append((s, 1))
            boundaries.append((e, -1))
    boundaries.sort(key=lambda b: (b[0], b[1]))
    active = 0
    max_active = 0
    parallel = 0.0
    prev_t = None
    for t, delta in boundaries:
        if prev_t is not None and active >= 2:
            parallel += t - prev_t
        active += delta
        max_active = max(max_active, active)
        prev_t = t
    return parallel, max_active

"""Interval arithmetic and turnaround summary KPIs."""
import pytest

from turnaround.analytics.intervals import clip_intervals, complement, concurrency_profile, merge_intervals, total_length
from turnaround.analytics.summary import compute_session_summary
from turnaround.domain.records import IntervalRecord, SessionRecord


def test_merge_clip_complement_total():
    assert merge_intervals([(5, 8), (0, 3), (2, 4), (4, 4.5), (9, 9)]) == [(0, 4.5), (5, 8)]
    assert total_length([(0, 4.5), (5, 8)]) == 7.5
    assert clip_intervals([(0, 10), (20, 30)], 5, 25) == [(5, 10), (20, 25)]
    assert complement((0, 20), [(2, 5), (8, 12)]) == [(0, 2), (5, 8), (12, 20)]
    assert complement((0, 10), []) == [(0, 10)]


def test_concurrency_profile():
    groups = [[(0, 10)], [(5, 15)], [(8, 9)]]
    parallel, peak = concurrency_profile(groups)
    assert peak == 3
    assert parallel == pytest.approx(5.0)  # only [5,10] has two or more groups active


def session(**kw):
    """Build a SessionRecord with overridable defaults (closed session, 0-100 s)."""
    base = dict(run_id="r", session_no=1, aircraft_id=1, first_seen_ts=0.0, end_ts=100.0, status="closed",
                closure_reason="aircraft_departed", partial_start=False, partial_end=False)
    base.update(kw)
    return SessionRecord(**base)


def iv(service, start, end, reason="released"):
    """Build an IntervalRecord (open when end is None)."""
    return IntervalRecord("r", 1, service, start, end, reason if end is not None else None)


def test_full_turnaround_kpis():
    ivs = [iv("aerobridge", 10, 60), iv("catering", 15, 30), iv("cargo", 20, 50), iv("pushback", 70, 80)]
    s = compute_session_summary(session(), ivs, now=100.0)
    assert s.window_start_ts == 10 and s.window_end_ts == 70  # pushback start ends the turnaround
    assert s.turnaround_time_s == 60
    assert s.arrival_to_first_service_s == 10
    assert s.pushback_to_departure_s == 30
    assert s.busy_s == 50 and s.idle_s == 10 and s.idle_gaps == [(60, 70)]
    assert s.utilization == pytest.approx(50 / 60)
    assert s.max_concurrent == 3
    assert s.parallel_s == pytest.approx(35.0)  # two or more services are connected during [15, 50]
    assert s.services["aerobridge"].total_s == 50 and s.services["pushback"].total_s == 10
    assert s.is_complete


def test_open_session_uses_now_and_marks_censored_and_incomplete():
    s = compute_session_summary(session(end_ts=None, status="open", closure_reason=None),
                                [iv("aerobridge", 10, None), iv("cargo", 20, 30, "shutdown")], now=90.0)
    assert s.reference_end_ts == 90.0 and s.session_duration_s == 90.0
    assert s.services["aerobridge"].is_open and s.services["aerobridge"].censored
    assert s.services["cargo"].censored and not s.services["cargo"].is_open
    assert s.window_end_ts == 90.0 and not s.is_complete


def test_no_services_and_partial_flags():
    s = compute_session_summary(session(), [], now=100.0)
    assert s.turnaround_time_s is None and s.utilization is None and not s.is_complete
    s = compute_session_summary(session(partial_start=True), [iv("pushback", 50, 60)], now=100.0)
    assert not s.is_complete and s.turnaround_time_s == 0


def test_same_service_reconnection_counts_and_merges():
    s = compute_session_summary(session(), [iv("catering", 10, 20), iv("catering", 18, 30), iv("catering", 40, 50)], now=100.0)
    st = s.services["catering"]
    assert st.count == 3 and st.total_s == 30 and st.first_start_ts == 10 and st.last_end_ts == 50

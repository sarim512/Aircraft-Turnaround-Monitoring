"""TurnaroundEngine end-to-end behaviour on synthetic 5 FPS scenarios."""
from turnaround.config.schema import AircraftConfig, AssociationConfig, RoiConfig, ServiceRuleConfig, TemporalConfig
from turnaround.domain.events import EndReason, EventType
from turnaround.domain.types import BoundingBox, ServiceType

from helpers import BASE_TS, FAR_BOX, aircraft, by_type, equipment, make_engine, run_script

AB = ServiceType.AEROBRIDGE


def present_always(t):
    """Frame function: one aircraft, no equipment."""
    return [aircraft()]


def test_flicker_and_short_occlusion_yield_one_clean_interval():
    engine = make_engine()

    def frames(t):
        dets = [aircraft()]
        in_service = 5.0 <= t < 45.0
        occluded = 20.0 <= t < 24.0  # 4 s < 8 s grace
        flicker_drop = int(round(t / 0.2)) % 4 == 0  # 25% missed detections
        if in_service and not occluded and not flicker_drop:
            dets.append(equipment(AB))
        return dets

    events = run_script(engine, 70.0, frames)
    started = by_type(events, EventType.SERVICE_STARTED, AB)
    ended = by_type(events, EventType.SERVICE_ENDED, AB)
    assert len(started) == 1 and len(ended) == 1
    assert by_type(events, EventType.SERVICE_DISCARDED) == []
    assert abs(started[0].ts - (BASE_TS + 5.0)) <= 0.41  # back-dated to first evidence
    assert abs(ended[0].ts - (BASE_TS + 44.6)) <= 0.41  # last evidence, not release time
    assert ended[0].payload["end_reason"] == EndReason.RELEASED.value


def test_occlusion_longer_than_grace_splits_into_two_intervals():
    engine = make_engine()

    def frames(t):
        dets = [aircraft()]
        if 5.0 <= t < 20.0 or 32.0 <= t < 50.0:  # 12 s hole > 8 s grace
            dets.append(equipment(AB))
        return dets

    events = run_script(engine, 70.0, frames)
    assert len(by_type(events, EventType.SERVICE_STARTED, AB)) == 2
    assert len(by_type(events, EventType.SERVICE_ENDED, AB)) == 2


def test_short_false_positive_burst_is_retracted():
    engine = make_engine()
    events = run_script(engine, 40.0, lambda t: [aircraft()] + ([equipment(AB)] if 10.0 <= t < 11.6 else []))
    assert len(by_type(events, EventType.SERVICE_STARTED, AB)) == 1
    discarded = by_type(events, EventType.SERVICE_DISCARDED, AB)
    assert len(discarded) == 1 and by_type(events, EventType.SERVICE_ENDED, AB) == []


def test_isolated_positives_never_start_a_service():
    engine = make_engine()
    events = run_script(engine, 60.0, lambda t: [aircraft()] + ([equipment(AB)] if int(round(t / 0.2)) % 15 == 0 else []))
    assert by_type(events, EventType.SERVICE_STARTED) == []


def test_equipment_without_aircraft_or_far_from_aircraft_is_ignored():
    engine = make_engine()
    events = run_script(engine, 30.0, lambda t: [equipment(AB)])
    assert events == []
    engine = make_engine()
    events = run_script(engine, 30.0, lambda t: [aircraft(), equipment(AB, box=FAR_BOX)])
    assert by_type(events, EventType.SERVICE_STARTED) == []


def test_tracker_id_switch_keeps_one_session():
    engine = make_engine()
    events = run_script(engine, 40.0, lambda t: [aircraft(track_id=1 if t < 20 else 9), equipment(AB)])
    assert len(by_type(events, EventType.SESSION_STARTED)) == 1
    assert by_type(events, EventType.SESSION_ENDED) == []


def test_departure_closes_session_and_clamps_open_interval():
    engine = make_engine()  # aircraft lost timeout 20 s
    events = run_script(engine, 70.0, lambda t: ([aircraft(), equipment(AB)] if t < 30.0 else []))
    ended_session = by_type(events, EventType.SESSION_ENDED)
    assert len(ended_session) == 1
    assert ended_session[0].payload["reason"] == EndReason.AIRCRAFT_DEPARTED.value
    assert ended_session[0].payload["partial_end"] is False
    assert abs(ended_session[0].ts - (BASE_TS + 29.8)) <= 0.21  # last sighting
    interval_end = by_type(events, EventType.SERVICE_ENDED, AB)[0]
    assert interval_end.ts <= ended_session[0].ts


def test_flush_closes_open_state_with_partial_end():
    engine = make_engine()
    events = run_script(engine, 30.0, lambda t: [aircraft(), equipment(AB)])
    flushed = engine.flush(BASE_TS + 30.0, EndReason.SHUTDOWN)
    assert [e.event_type for e in flushed] == [EventType.SERVICE_ENDED, EventType.SESSION_ENDED]
    assert flushed[0].payload["end_reason"] == "shutdown" and flushed[1].payload["partial_end"] is True
    assert engine.flush(BASE_TS + 31.0, EndReason.SHUTDOWN) == []


def test_partial_start_flag():
    engine = make_engine()
    events = run_script(engine, 20.0, present_always)
    assert by_type(events, EventType.SESSION_STARTED)[0].payload["partial_start"] is True
    engine = make_engine()
    events = run_script(engine, 60.0, lambda t: [aircraft()] if t >= 30.0 else [])
    assert by_type(events, EventType.SESSION_STARTED)[0].payload["partial_start"] is False


def test_long_stream_stall_does_not_release_active_service():
    engine = make_engine()
    stall = lambda t: 20.0 <= t < 40.0  # no frames at all for 20 s (credited as <= 3 s)
    events = run_script(engine, 60.0, lambda t: [aircraft(), equipment(AB)] if t < 20.0 else [aircraft()], skip=stall)
    # after resuming only 20 s - but frames after 40 s have no equipment: absence accrues at real 0.2 s steps
    ended = by_type(events, EventType.SERVICE_ENDED, AB)
    assert len(ended) == 1
    assert ended[0].ts < BASE_TS + 20.0  # last evidence before the stall


def test_roi_analysis_zone_blocks_outside_equipment():
    roi = RoiConfig(enabled=True, analysis_zone=[[0, 0], [0.5, 0], [0.5, 1], [0, 1]])  # left half only
    engine = make_engine(roi_cfg=roi)
    events = run_script(engine, 30.0, lambda t: [aircraft(BoundingBox(20, 100, 300, 300)), equipment(AB, box=BoundingBox(400, 150, 460, 250))])
    assert by_type(events, EventType.SERVICE_STARTED) == []


def test_per_service_rule_override_applies():
    temporal = TemporalConfig(
        defaults=ServiceRuleConfig(on_window_s=3.0, on_min_hits=4, off_grace_s=8.0, min_interval_s=5.0),
        overrides={"pushback": {"off_grace_s": 2.0}},
    )
    engine = make_engine(temporal=temporal)
    PB = ServiceType.PUSHBACK
    events = run_script(engine, 40.0, lambda t: [aircraft()] + ([equipment(PB), equipment(AB)] if 5.0 <= t < 20.0 else []))
    end_pb = by_type(events, EventType.SERVICE_ENDED, PB)[0]
    end_ab = by_type(events, EventType.SERVICE_ENDED, AB)[0]
    assert end_pb.payload["decided_at"] < end_ab.payload["decided_at"]  # shorter grace releases first


def test_two_aircraft_get_independent_sessions():
    engine = make_engine()
    other = BoundingBox(700, 100, 1100, 300)
    events = run_script(engine, 30.0, lambda t: [aircraft(track_id=1), aircraft(other, track_id=2), equipment(AB)])
    sessions = by_type(events, EventType.SESSION_STARTED)
    assert len(sessions) == 2
    assert {e.aircraft_id for e in by_type(events, EventType.SERVICE_STARTED, AB)} == {1}

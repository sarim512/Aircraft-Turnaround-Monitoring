"""AircraftRegistry: confirmation, id switches, coasting, departure, duplicates."""
import pytest

from turnaround.config.schema import AircraftConfig
from turnaround.domain.types import BoundingBox
from turnaround.temporal.aircraft_registry import AircraftRegistry

from helpers import AIRCRAFT_BOX, aircraft

CFG = AircraftConfig(min_confidence=0.4, confirm_hits=3, tentative_timeout_s=3.0, lost_timeout_s=10.0)


def step(reg, t, dets, dt=0.2):
    """Advance the registry by one 5 FPS frame (dt = 0.2 s unless given)."""
    return reg.update(t, dt, dets)


def test_confirms_after_confirm_hits_and_reports_once():
    reg = AircraftRegistry(CFG)
    assert step(reg, 0.0, [aircraft()]).snapshots == []
    assert step(reg, 0.2, [aircraft()]).newly_confirmed == []
    up = step(reg, 0.4, [aircraft()])
    assert len(up.newly_confirmed) == 1 and len(up.snapshots) == 1
    assert up.snapshots[0].first_seen_ts == 0.0 and up.snapshots[0].observed
    assert step(reg, 0.6, [aircraft()]).newly_confirmed == []


def test_low_confidence_ignored():
    reg = AircraftRegistry(CFG)
    for i in range(5):
        assert step(reg, i * 0.2, [aircraft(conf=0.2)]).snapshots == []


def test_tracker_id_switch_keeps_same_aircraft_via_iou():
    reg = AircraftRegistry(CFG)
    for i in range(4):
        up = step(reg, i * 0.2, [aircraft(track_id=1)])
    first_id = up.snapshots[0].aircraft_id
    up = step(reg, 1.0, [aircraft(track_id=7)])  # ByteTrack assigned a new id
    assert [s.aircraft_id for s in up.snapshots] == [first_id]
    up = step(reg, 1.2, [aircraft(track_id=7)])
    assert [s.aircraft_id for s in up.snapshots] == [first_id]


def test_coasts_through_occlusion_then_resumes_same_identity():
    reg = AircraftRegistry(CFG)
    for i in range(4):
        up = step(reg, i * 0.2, [aircraft()])
    aid = up.snapshots[0].aircraft_id
    for i in range(10):  # 2 s without detection
        up = step(reg, 1.0 + i * 0.2, [])
        assert len(up.snapshots) == 1 and not up.snapshots[0].observed and up.departed == []
    up = step(reg, 3.0, [aircraft(track_id=3)])
    assert up.snapshots[0].aircraft_id == aid and up.snapshots[0].observed


def test_departs_after_lost_timeout_with_last_seen_ts():
    reg = AircraftRegistry(CFG)
    for i in range(4):
        step(reg, i * 0.2, [aircraft()])
    departed = []
    for i in range(1, 80):
        up = step(reg, 0.6 + i * 0.2, [])
        departed += up.departed
    assert len(departed) == 1 and departed[0].last_seen_ts == pytest.approx(0.6)
    assert reg.confirmed_ids() == []


def test_tentative_slot_dropped_without_confirmation():
    reg = AircraftRegistry(CFG)
    step(reg, 0.0, [aircraft()])
    for i in range(1, 30):
        up = step(reg, i * 0.2, [])
        assert up.newly_confirmed == []
    step(reg, 6.2, [aircraft()])
    step(reg, 6.4, [aircraft()])
    assert step(reg, 6.6, [aircraft()]).newly_confirmed  # fresh slot needed 3 hits again


def test_duplicate_boxes_are_merged_and_far_aircraft_get_separate_ids():
    reg = AircraftRegistry(CFG)
    near_dup = BoundingBox(102, 101, 501, 301)
    far = BoundingBox(900, 100, 1300, 300)
    for i in range(4):
        up = step(reg, i * 0.2, [aircraft(track_id=1), aircraft(near_dup, conf=0.6, track_id=2), aircraft(far, track_id=3)])
    assert len(up.snapshots) == 2
    assert len({s.aircraft_id for s in up.snapshots}) == 2

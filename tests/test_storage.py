"""SQLite writer projections and read-only reader."""
import sqlite3

import pytest

from turnaround.domain.events import DomainEvent, EndReason, EventType
from turnaround.domain.types import RunInfo, ServiceType
from turnaround.storage.reader import AnalyticsReader
from turnaround.storage.writer import SqliteEventWriter

AB = ServiceType.AEROBRIDGE


@pytest.fixture
def db(tmp_path):
    """Fixture: (db path, SqliteEventWriter with an open run ``run1``)."""
    path = str(tmp_path / "sub" / "t.db")
    writer = SqliteEventWriter(path)
    writer.begin_run(RunInfo("run1", 1.0, "video.mp4", {"a": 1}))
    yield path, writer
    writer.close()


def ev(event_type, ts, service=None, **payload):
    """Build a DomainEvent for session 1 / aircraft 1."""
    return DomainEvent(event_type, ts, session_no=1, aircraft_id=1, service=service, payload=payload)


def test_projection_roundtrip(db):
    path, w = db
    w.handle([ev(EventType.SESSION_STARTED, 10.0, partial_start=True), ev(EventType.SERVICE_STARTED, 12.0, AB, decided_at=13.0)])
    r = AnalyticsReader(path)
    assert r.list_intervals("run1")[0].end_ts is None
    assert r.list_sessions("run1")[0].status == "open" and r.list_sessions("run1")[0].partial_start
    w.handle([ev(EventType.SERVICE_ENDED, 40.0, AB, end_reason="released"), ev(EventType.SESSION_ENDED, 50.0, reason="aircraft_departed", partial_end=False)])
    iv = r.list_intervals("run1")[0]
    s = r.list_sessions("run1")[0]
    assert (iv.start_ts, iv.end_ts, iv.end_reason) == (12.0, 40.0, "released")
    assert (s.status, s.end_ts, s.closure_reason, s.partial_end) == ("closed", 50.0, "aircraft_departed", False)
    events = r.list_events("run1")
    assert len(events) == 4 and events[0].event_type == "session_ended" and events[-1].payload["partial_start"] is True
    assert r.latest_event_ts("run1") == 50.0 and r.latest_run_id() == "run1"


def test_discarded_interval_is_removed_but_event_kept(db):
    path, w = db
    w.handle([ev(EventType.SESSION_STARTED, 10.0), ev(EventType.SERVICE_STARTED, 12.0, AB)])
    w.handle([ev(EventType.SERVICE_DISCARDED, 13.0, AB, reason="shorter_than_min_interval")])
    r = AnalyticsReader(path)
    assert r.list_intervals("run1") == []
    assert [e.event_type for e in r.list_events("run1")].count("service_discarded") == 1


def test_scene_events_without_session_and_event_filters(db):
    path, w = db
    w.handle([DomainEvent(EventType.SCENE_CHANGED, 5.0, payload={"similarity": 0.1}), ev(EventType.SESSION_STARTED, 6.0)])
    r = AnalyticsReader(path)
    assert len(r.list_events("run1")) == 2 and len(r.list_events("run1", session_no=1)) == 1


def test_reader_survives_missing_db_and_failed_batch_rolls_back(tmp_path, db):
    assert AnalyticsReader(str(tmp_path / "none.db")).list_runs() == []
    path, w = db
    w.handle([ev(EventType.SESSION_STARTED, 10.0)])
    bad_batch = [ev(EventType.SESSION_STARTED, 20.0), DomainEvent(EventType.SCENE_CHANGED, None)]  # NULL ts violates NOT NULL
    with pytest.raises(sqlite3.Error):
        w.handle(bad_batch)
    assert len(AnalyticsReader(path).list_events("run1")) == 1  # whole failed batch rolled back


def test_end_run_and_handle_before_begin(tmp_path):
    w = SqliteEventWriter(str(tmp_path / "x.db"))
    with pytest.raises(RuntimeError):
        w.handle([ev(EventType.SESSION_STARTED, 1.0)])
    w.begin_run(RunInfo("r", 1.0, "s", {}))
    w.end_run(99.0)
    assert AnalyticsReader(str(tmp_path / "x.db")).list_runs()[0].ended_ts == 99.0
    w.close()

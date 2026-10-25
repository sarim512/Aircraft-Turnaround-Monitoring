"""Dashboard smoke tests (Streamlit AppTest) with a populated and an empty database."""
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from turnaround.dashboard import charts, data_access
from turnaround.analytics.summary import compute_session_summary
from turnaround.pipeline.builder import build_pipeline
from turnaround.storage.reader import AnalyticsReader

from test_pipeline_integration import ScriptedTracker, make_config, make_video, turnaround_script

APP = str(Path(__file__).parents[1] / "src" / "turnaround" / "dashboard" / "app.py")


def write_config(cfg, tmp_path):
    """Dump an AppConfig as a YAML file the dashboard can load."""
    path = tmp_path / "dash.yaml"
    path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    return str(path)


@pytest.fixture
def populated(tmp_path):
    """Fixture: run a full synthetic pipeline run and return (config, yaml path)."""
    cfg = make_config(tmp_path, make_video(tmp_path / "v.avi", seconds=70))
    build_pipeline(cfg, tracker=ScriptedTracker(turnaround_script)).run()
    return cfg, write_config(cfg, tmp_path)


def test_dashboard_renders_populated_run(populated, monkeypatch):
    cfg, cfg_path = populated
    monkeypatch.setenv("TURNAROUND_CONFIG", cfg_path)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Sessions" in labels and "Mean turnaround" in labels
    assert any(df.value.shape[0] == 1 for df in at.dataframe)  # one session row
    assert at.title[0].value == "Aircraft turnaround monitor"


def test_dashboard_handles_missing_database(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, str(tmp_path / "v.avi"))
    monkeypatch.setenv("TURNAROUND_CONFIG", write_config(cfg, tmp_path))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("No runs recorded" in i.value for i in at.info)


def test_data_access_and_charts(populated):
    cfg, _ = populated
    reader = AnalyticsReader(cfg.storage.db_path)
    run_id = reader.latest_run_id()
    now = data_access.resolve_now(reader, None, run_id)
    summaries = data_access.compute_summaries(reader, run_id, now)
    frame = data_access.summaries_to_frame(summaries)
    assert list(frame["Aircraft"]) == ["A1"] and frame["Reliable"].iloc[0]
    assert data_access.overview(summaries)["reliable_sessions"] == 1.0
    intervals = reader.list_intervals(run_id)
    gantt = data_access.gantt_frame(intervals, summaries[0])
    assert set(gantt["service"]) == {"aerobridge", "catering", "pushback"}
    assert len(charts.gantt_figure(gantt, summaries[0]).data) >= 1
    assert len(charts.service_duration_figure(summaries[0]).data) == 1
    assert data_access.format_duration(3725) == "1:02:05" and data_access.format_duration(None) == "-"
    assert not data_access.events_to_frame(reader.list_events(run_id)).empty

"""Streamlit dashboard: live view, turnaround analytics and event log.

Run with ``python scripts/run_dashboard.py`` (sets TURNAROUND_CONFIG) or
``streamlit run src/turnaround/dashboard/app.py``.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd
import streamlit as st

from turnaround.analytics.summary import TurnaroundSummary
from turnaround.config.loader import load_config
from turnaround.config.schema import AppConfig
from turnaround.dashboard import charts, data_access
from turnaround.runtime.live_state import LiveStateReader
from turnaround.storage.reader import AnalyticsReader

CONFIG_ENV = "TURNAROUND_CONFIG"
STALE_AFTER_S = 15.0


@st.cache_resource
def _load_config(path: Optional[str]) -> AppConfig:
    """Load (and cache) the configuration.

    Args:
        path: YAML path from ``TURNAROUND_CONFIG`` or None for the default.
    Returns:
        AppConfig.
    """
    return load_config(path)


def _stretch() -> Dict[str, Any]:
    """Keyword argument that makes a widget fill the container, across Streamlit versions.

    Returns:
        ``{"width": "stretch"}`` on Streamlit >= 1.50, else ``{"use_container_width": True}``
        (also the fallback when the version string cannot be parsed).
    """
    try:
        major, minor = (int(part) for part in st.__version__.split(".")[:2])
    except ValueError:
        return {"use_container_width": True}
    return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}


def _fragment(refresh_s: Optional[float]):
    """Return a decorator that re-runs a panel periodically.

    Args:
        refresh_s: Refresh period in seconds, or None for no auto-refresh.
    Returns:
        ``st.fragment`` decorator (with ``run_every`` when refresh_s is set).
    """
    return st.fragment(run_every=refresh_s) if refresh_s else st.fragment


def _status_badge(status: Optional[Mapping[str, Any]]) -> str:
    """Human-readable pipeline state line.

    Args:
        status: Parsed status.json or None.
    Returns:
        Short sentence describing the pipeline state (no pipeline / stale / state label).
    """
    if not status:
        return "No pipeline status yet"
    age = time.time() - float(status.get("updated_wall_ts", 0))
    if status.get("state") != "finished" and age > STALE_AFTER_S:
        return f"Pipeline silent for {int(age)} s"
    labels = {
        "running": "Running",
        "paused_scene_change": "Paused: camera view changed",
        "waiting_for_frames": "Waiting for frames",
        "finished": "Finished",
    }
    return labels.get(str(status.get("state")), str(status.get("state")))


def _live_panel(live: LiveStateReader, reader: AnalyticsReader, run_id: Optional[str]) -> None:
    """Render the live frame and pipeline metrics.

    Args:
        live: Live-state reader.
        reader: Analytics reader (unused except for symmetry with other panels).
        run_id: Selected run (status of other runs is flagged).
    Logic:
        Read status.json. Left column: latest.jpg, or an info box explaining how to start the pipeline. Right
        column: state heading; a note when the live feed belongs to another run; metrics for fps, latency and
        processed frames; a reconnect/drop caption when the source reports reconnects; then one line per open
        session listing the connected services (or "no equipment connected").
    """
    status = live.read_status()
    left, right = st.columns([2, 1])
    with left:
        frame = live.read_frame_bytes()
        if frame:
            st.image(frame, **_stretch())
        else:
            st.info("No frame yet. Start the pipeline with `python scripts/run_pipeline.py` and the live view will appear here.")
    with right:
        st.subheader(_status_badge(status))
        if status:
            if run_id and status.get("run_id") != run_id:
                st.caption("The live feed belongs to a different run than the one selected in the sidebar.")
            c1, c2 = st.columns(2)
            c1.metric("Frames per second", status.get("fps", 0))
            c2.metric("Latency (ms)", status.get("latency_ms", 0))
            st.metric("Frames processed", status.get("frames_processed", 0))
            stats: Dict[str, Any] = status.get("source_stats") or {}
            if stats.get("reconnects"):
                st.caption(f"Stream reconnects: {int(stats['reconnects'])}; dropped frames: {int(stats.get('frames_dropped', 0))}")
            st.markdown("**Aircraft on stand**")
            sessions: List[Dict[str, Any]] = status.get("sessions") or []
            if not sessions:
                st.caption("No aircraft confirmed yet.")
            for s in sessions:
                services = ", ".join(s["active_services"]) or "no equipment connected"
                st.write(f"Session {s['session_no']} (A{s['aircraft_id']}): {services}")


def _analytics_panel(reader: AnalyticsReader, live: LiveStateReader, run_id: Optional[str]) -> None:
    """Render KPIs, the sessions table and the per-session timeline.

    Args:
        reader: Analytics reader.
        live: Live-state reader (supplies the stream clock for open sessions).
        run_id: Selected run id.
    Logic:
        Show an info message when there is no run or no session. Resolve the stream-clock "now", compute all
        summaries, show four headline metrics (sessions, open now, mean/median turnaround of reliable sessions),
        an explanatory caption and the sessions table. A selectbox picks one session: fetch its intervals (info
        message if none), draw the Gantt chart and per-service bars, show per-session metrics (turnaround, arrival
        to first connection, idle time, parallel time) and a table of idle gaps when present.
    """
    if not run_id:
        st.info("No runs recorded yet.")
        return
    now = data_access.resolve_now(reader, live.read_status(), run_id)
    summaries: List[TurnaroundSummary] = data_access.compute_summaries(reader, run_id, now)
    if not summaries:
        st.info("No turnaround sessions in this run yet. A session opens once an aircraft is confirmed.")
        return
    ov = data_access.overview(summaries)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sessions", int(ov["sessions"] or 0), help="All sessions of the selected run")
    k2.metric("Open now", int(ov["open_sessions"] or 0))
    k3.metric("Mean turnaround", data_access.format_duration(ov["mean_turnaround_s"]), help="Reliable sessions only")
    k4.metric("Median turnaround", data_access.format_duration(ov["median_turnaround_s"]), help="Reliable sessions only")
    st.caption(
        "Reliable = closed session with a pushback, observed from arrival to departure. "
        "Turnaround runs from the first equipment connection to the start of pushback (or the last disconnection)."
    )
    st.dataframe(data_access.summaries_to_frame(summaries), **_stretch(), hide_index=True)

    options = {f"Session {s.session_no} (A{s.aircraft_id}, {s.status})": s for s in summaries}
    choice = st.selectbox("Session detail", list(options))
    summary = options[choice]
    intervals = [iv for iv in reader.list_intervals(run_id) if iv.session_no == summary.session_no]
    if not intervals:
        st.info("No equipment connection recorded for this session yet.")
        return
    st.plotly_chart(charts.gantt_figure(data_access.gantt_frame(intervals, summary), summary), **_stretch())
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(charts.service_duration_figure(summary), **_stretch())
    with right:
        st.metric("Turnaround", data_access.format_duration(summary.turnaround_time_s))
        st.metric("Arrival to first connection", data_access.format_duration(summary.arrival_to_first_service_s))
        st.metric("Idle inside turnaround", data_access.format_duration(summary.idle_s))
        st.metric("Time with two or more services", data_access.format_duration(summary.parallel_s))
        if summary.idle_gaps:
            gaps = pd.DataFrame(
                [{"From": data_access.to_datetime(a), "To": data_access.to_datetime(b), "Length": data_access.format_duration(b - a)} for a, b in summary.idle_gaps]
            )
            st.caption("Idle gaps")
            st.dataframe(gaps, hide_index=True, **_stretch())


def _events_panel(reader: AnalyticsReader, run_id: Optional[str]) -> None:
    """Render the filterable event log.

    Args:
        reader: Analytics reader.
        run_id: Selected run id.
    """
    if not run_id:
        st.info("No runs recorded yet.")
        return
    frame = data_access.events_to_frame(reader.list_events(run_id, limit=1000))
    if frame.empty:
        st.info("No events recorded for this run yet.")
        return
    types = st.multiselect("Event types", sorted(frame["Event"].unique()), default=sorted(frame["Event"].unique()))
    st.dataframe(frame[frame["Event"].isin(types)], **_stretch(), hide_index=True)


def main() -> None:
    """Build the page: sidebar (run + refresh), then the three tabs.

    Logic:
        Load the config named by ``TURNAROUND_CONFIG``; build the readers; sidebar offers run selection
        (default: newest) and the auto-refresh period (0 disables); each tab renders its panel through a
        fragment so only that panel reruns on the refresh timer.
    """
    st.set_page_config(page_title="Turnaround monitor", layout="wide")
    cfg = _load_config(os.environ.get(CONFIG_ENV))
    reader = AnalyticsReader(cfg.storage.db_path)
    live = LiveStateReader(cfg.output.runtime_dir)

    st.title("Aircraft turnaround monitor")
    with st.sidebar:
        st.header("Run")
        runs = reader.list_runs()
        labels = {f"{r.run_id}  ({r.source_uri[-40:]})": r.run_id for r in runs}
        run_id = labels[st.selectbox("Recorded run", list(labels))] if labels else None
        refresh = st.slider("Refresh every (seconds, 0 = off)", 0, 30, 3)
        st.caption(f"Database: {cfg.storage.db_path}")

    every = float(refresh) if refresh else None
    tab_live, tab_analytics, tab_events = st.tabs(["Live", "Turnaround analytics", "Event log"])
    with tab_live:
        _fragment(every)(_live_panel)(live, reader, run_id)
    with tab_analytics:
        _fragment(every)(_analytics_panel)(reader, live, run_id)
    with tab_events:
        _fragment(every)(_events_panel)(reader, run_id)


if __name__ == "__main__":
    main()

"""Plotly figures for the dashboard (pure functions returning figures)."""
from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from turnaround.analytics.summary import TurnaroundSummary
from turnaround.dashboard.data_access import to_datetime
from turnaround.domain.types import SERVICE_NAMES

SERVICE_HEX: Dict[str, str] = {
    "aerobridge": "#3DB5C9",
    "catering": "#7BC47F",
    "cargo": "#F0A04B",
    "pushback": "#E5605A",
}
PLOT_BG = "rgba(0,0,0,0)"


def gantt_figure(frame: pd.DataFrame, summary: TurnaroundSummary) -> go.Figure:
    """Timeline of service intervals with idle gaps shaded.

    Args:
        frame: Output of ``data_access.gantt_frame``.
        summary: Session summary (supplies idle gaps and the session bounds).
    Returns:
        Plotly figure with one row per service (canonical order), idle gaps as gray bands inside the
        service window, and thin lines marking the first sighting and the session end.
    Logic:
        Build ``px.timeline`` (service colors, canonical row order, hover details); reverse the y axis; add a
        translucent ``vrect`` behind each idle gap; draw dotted vertical lines (paper-relative shapes, avoiding
        annotation problems on datetime axes) at the first sighting and, if closed, the session end; hide the
        legend, make the background transparent and fix a compact height.
    """
    fig = px.timeline(
        frame,
        x_start="start",
        x_end="end",
        y="service",
        color="service",
        color_discrete_map=SERVICE_HEX,
        category_orders={"service": list(SERVICE_NAMES)},
        hover_data={"duration_s": ":.0f", "state": True, "end_reason": True},
    )
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title=None)
    for start, end in summary.idle_gaps:
        fig.add_vrect(x0=to_datetime(start), x1=to_datetime(end), fillcolor="rgba(160,170,180,0.18)", line_width=0, layer="below")
    marks = [summary.first_seen_ts] + ([summary.end_ts] if summary.end_ts is not None else [])
    for ts in marks:
        x = to_datetime(ts)
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper", line=dict(color="rgba(228,234,238,0.45)", width=1, dash="dot"))
    fig.update_layout(showlegend=False, height=280, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG)
    return fig


def service_duration_figure(summary: TurnaroundSummary) -> go.Figure:
    """Horizontal bars of total connected minutes per service.

    Args:
        summary: Session summary.
    Returns:
        Plotly bar figure (services in canonical order).
    """
    minutes = [round(summary.services[name].total_s / 60.0, 1) for name in SERVICE_NAMES]
    fig = go.Figure(go.Bar(x=minutes, y=list(SERVICE_NAMES), orientation="h", marker_color=[SERVICE_HEX[n] for n in SERVICE_NAMES]))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="minutes connected")
    fig.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG)
    return fig

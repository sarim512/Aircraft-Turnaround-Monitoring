"""Builds the right FrameSource from a SourceConfig (the only place that knows all source kinds)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from turnaround.config.schema import SourceConfig
from turnaround.ingestion.base import FrameSource
from turnaround.ingestion.file_source import VideoFileSource
from turnaround.ingestion.stream_source import StreamFrameSource
from turnaround.ingestion.youtube import resolve_youtube_stream


def build_source(cfg: SourceConfig) -> FrameSource:
    """Create a frame source for the configured kind.

    Args:
        cfg: Validated SourceConfig.
    Returns:
        A not-yet-started FrameSource.
    Raises:
        ValueError: Unknown ``cfg.kind`` or malformed ``start_time_iso``.
    Logic:
        ``youtube`` -> StreamFrameSource whose resolver calls ``resolve_youtube_stream`` (re-resolved on
        every reconnect); ``stream`` -> StreamFrameSource with a constant resolver; ``file`` ->
        VideoFileSource with the optional ISO start time converted to epoch seconds.
    """
    if cfg.kind == "youtube":
        return StreamFrameSource(
            resolver=lambda: resolve_youtube_stream(cfg.uri, cfg.max_height),
            target_fps=cfg.target_fps,
            buffer_size=cfg.buffer_size,
            reconnect_initial_s=cfg.reconnect_initial_s,
            reconnect_max_s=cfg.reconnect_max_s,
            source_id=cfg.uri,
            read_timeout_s=cfg.read_timeout_s,
        )
    if cfg.kind == "stream":
        return StreamFrameSource(
            resolver=lambda: cfg.uri,
            target_fps=cfg.target_fps,
            buffer_size=cfg.buffer_size,
            reconnect_initial_s=cfg.reconnect_initial_s,
            reconnect_max_s=cfg.reconnect_max_s,
            source_id=cfg.uri,
            read_timeout_s=cfg.read_timeout_s,
        )
    if cfg.kind == "file":
        start_epoch: Optional[float] = None
        if cfg.start_time_iso:
            start_epoch = datetime.fromisoformat(cfg.start_time_iso).timestamp()
        return VideoFileSource(
            path=cfg.uri,
            target_fps=cfg.target_fps,
            realtime_pacing=cfg.realtime_pacing,
            loop=cfg.loop,
            start_epoch=start_epoch,
        )
    raise ValueError(f"Unknown source kind: {cfg.kind!r}")

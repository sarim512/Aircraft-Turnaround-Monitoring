"""Command-line entry point: run the turnaround monitoring pipeline."""
import _bootstrap  # noqa: F401  (adds src/ to sys.path)

import argparse
import logging
import signal
import sys
from typing import List, Optional

from turnaround.config.loader import ConfigError, load_config
from turnaround.pipeline.builder import build_pipeline
from turnaround.utils.logging_setup import configure_logging


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line options.

    Args:
        argv: Argument list (None uses ``sys.argv[1:]``).
    Returns:
        Namespace with config, source, kind, weights, device, db, max_frames, log_level.
    """
    p = argparse.ArgumentParser(description="Aircraft turnaround monitoring pipeline")
    p.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    p.add_argument("--source", default=None, help="Override source.uri (YouTube URL, stream URL or video file)")
    p.add_argument("--kind", choices=["youtube", "stream", "file"], default=None, help="Override source.kind")
    p.add_argument("--weights", default=None, help="Override model.weights_path")
    p.add_argument("--device", default=None, help="Override model.device (auto|cpu|cuda:0)")
    p.add_argument("--db", default=None, help="Override storage.db_path")
    p.add_argument("--max-frames", type=int, default=None, help="Stop after this many processed frames")
    p.add_argument("--log-level", default=None, help="Override logging.level")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Load config, build the pipeline, install signal handlers and run.

    Args:
        argv: Argument list (None uses ``sys.argv[1:]``).
    Returns:
        Process exit code (0 success, 2 configuration/setup error, 3 frame source could not start).
    Logic:
        1. Translate CLI flags into dotted config overrides and ``load_config``.
        2. Configure logging from the config.
        3. ``build_pipeline`` (loads weights, validates classes) - setup errors are logged and return 2.
        4. SIGINT/SIGTERM call ``runner.request_stop`` for a graceful flush.
        5. ``runner.run``, log the final statistics and return 3 if the source could not start.
    """
    args = parse_args(argv)
    overrides = {
        "source.uri": args.source,
        "source.kind": args.kind,
        "model.weights_path": args.weights,
        "model.device": args.device,
        "storage.db_path": args.db,
        "logging.level": args.log_level,
    }
    try:
        cfg = load_config(args.config, overrides)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    configure_logging(cfg.logging.level, cfg.logging.file)
    log = logging.getLogger("run_pipeline")
    try:
        runner = build_pipeline(cfg)
    except Exception as exc:
        log.error("Pipeline setup failed: %s", exc)
        return 2
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: runner.request_stop())
    stats = runner.run(max_frames=args.max_frames)
    log.info("Done: %s", stats)
    return 3 if stats.stop_reason == "source_error" else 0


if __name__ == "__main__":
    sys.exit(main())

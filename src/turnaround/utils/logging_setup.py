"""Logging configuration shared by all entry points."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure the root logger (idempotent).

    Args:
        level: Logging level name.
        log_file: Optional file path; adds a rotating file handler (5 MB x 3 backups).
    Side effects:
        Replaces existing root handlers; creates the log file's parent directory.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

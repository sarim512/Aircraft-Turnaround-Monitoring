"""SQLite connection helpers shared by the writer and the reader."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(db_path: Union[str, Path], read_only: bool = False) -> sqlite3.Connection:
    """Open a SQLite connection.

    Args:
        db_path: Database file path.
        read_only: Open with ``mode=ro`` (used by the dashboard; requires the file to exist).
    Returns:
        Connection with ``sqlite3.Row`` rows. Writers get WAL journaling, ``synchronous=NORMAL`` and
        foreign keys on; the parent directory is created.
    Raises:
        sqlite3.OperationalError: Read-only open of a missing file or locked database after 10 s.
    Side effects:
        Creates the parent directory and the database file for write connections.
    """
    path = Path(db_path)
    if read_only:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=10.0)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes if they do not exist.

    Args:
        conn: Writable connection.
    Side effects:
        Executes ``schema.sql`` and commits.
    """
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()

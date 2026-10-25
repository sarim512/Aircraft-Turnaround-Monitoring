-- Aircraft turnaround monitor: SQLite schema (idempotent).
-- Timestamps are REAL epoch seconds on the *stream clock* (wall clock for live, video timeline for replay).

CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    started_ts   REAL NOT NULL,
    ended_ts     REAL,
    source_uri   TEXT NOT NULL,
    config_json  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    run_id         TEXT    NOT NULL REFERENCES runs(run_id),
    session_no     INTEGER NOT NULL,
    aircraft_id    INTEGER NOT NULL,
    first_seen_ts  REAL    NOT NULL,
    end_ts         REAL,
    status         TEXT    NOT NULL CHECK (status IN ('open', 'closed')),
    closure_reason TEXT,
    partial_start  INTEGER NOT NULL DEFAULT 0,
    partial_end    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, session_no)
);

CREATE TABLE IF NOT EXISTS service_intervals (
    run_id      TEXT    NOT NULL,
    session_no  INTEGER NOT NULL,
    service     TEXT    NOT NULL,
    start_ts    REAL    NOT NULL,
    end_ts      REAL,
    end_reason  TEXT,
    PRIMARY KEY (run_id, session_no, service, start_ts),
    FOREIGN KEY (run_id, session_no) REFERENCES sessions(run_id, session_no)
);

CREATE TABLE IF NOT EXISTS events (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    ts           REAL NOT NULL,
    event_type   TEXT NOT NULL,
    session_no   INTEGER,
    aircraft_id  INTEGER,
    service      TEXT,
    payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_run_ts ON events (run_id, ts);
CREATE INDEX IF NOT EXISTS idx_intervals_session ON service_intervals (run_id, session_no);
CREATE INDEX IF NOT EXISTS idx_sessions_run ON sessions (run_id, first_seen_ts);

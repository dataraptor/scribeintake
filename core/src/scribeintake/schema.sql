-- ScribeIntake SQLite schema (spec section 13).
--
-- v1 has no migration framework: the schema is created from this file and
-- recreated on change during development (synthetic data only, so drop-and-recreate
-- is safe). JSON is stored as TEXT; timestamps are ISO-8601 TEXT (SQLite has no
-- native datetime). All tables use CREATE TABLE IF NOT EXISTS so init_db is idempotent.

CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    started_at   TEXT NOT NULL,
    completed_at TEXT,
    status       TEXT NOT NULL DEFAULT 'active',
    triage_band  TEXT,                                  -- final predicted band
    triage_floor TEXT NOT NULL DEFAULT 'self_care',     -- monotonic safety floor
    floor_pinned INTEGER NOT NULL DEFAULT 0,            -- 0/1 boolean
    signals_json TEXT,                                  -- latest Signals snapshot
    language     TEXT NOT NULL DEFAULT 'en-US'
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role       TEXT NOT NULL,                           -- user | assistant
    content    TEXT NOT NULL,
    model      TEXT,                                    -- which model produced an assistant turn
    ts         TEXT NOT NULL
);

-- Append-only audit of slot writes (latest-wins on read by max id per slot).
CREATE TABLE IF NOT EXISTS intake_state (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(id),
    slot          TEXT NOT NULL,
    value         TEXT,
    confidence    TEXT,
    source_msg_id TEXT,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    soap_json  TEXT NOT NULL,
    version    TEXT NOT NULL,
    ts         TEXT NOT NULL
);

-- Doubles as the observability/audit log. The two cache token columns + cost_usd
-- make cost accounting honest (spec section 16); prompt/rules versions make eval
-- runs reproducible (spec section 13).
CREATE TABLE IF NOT EXISTS tool_calls (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id            TEXT NOT NULL REFERENCES sessions(id),
    turn                  INTEGER,
    tool                  TEXT NOT NULL,
    args_json             TEXT,
    result_json           TEXT,
    input_tokens          INTEGER NOT NULL DEFAULT 0,
    output_tokens         INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens     INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms            INTEGER,
    model                 TEXT,
    cost_usd              REAL NOT NULL DEFAULT 0.0,
    prompt_version        TEXT,
    rules_version         TEXT,
    ts                    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS safety_events (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT NOT NULL REFERENCES sessions(id),
    level              TEXT NOT NULL,                   -- CLEAR | URGENT | EMERGENCY
    source             TEXT NOT NULL,                   -- gate | agent
    matched_rules_json TEXT,
    rules_version      TEXT,
    msg_id             TEXT,
    ts                 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kb_chunks (
    id            TEXT PRIMARY KEY,
    source        TEXT,
    url           TEXT,
    license       TEXT,
    jurisdiction  TEXT,
    section       TEXT,
    last_reviewed TEXT,
    text          TEXT NOT NULL,
    embedding     BLOB
);

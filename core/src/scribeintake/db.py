"""SQLite access layer — stateless per turn.

Every turn loads session state from SQLite, runs, and writes back; there is **no
module-level mutable session cache**. This is what makes eval runs isolated and
parallel-safe (spec section 6). Helpers round-trip through the Pydantic models in
:mod:`scribeintake.models`. Later splits extend this module; keep it small.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

from .config import settings
from .models import (
    Confidence,
    IntakeState,
    Signals,
    SlotValue,
    ToolCallTrace,
    TriageBand,
)

SCHEMA_SQL = resources.files("scribeintake").joinpath("schema.sql").read_text(encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ------------------------------------------------------------------- connection
def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a connection with ``Row`` factory and foreign keys enabled."""
    path = Path(db_path) if db_path is not None else settings.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables from ``schema.sql`` (idempotent)."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def reset_db(db_path: str | Path) -> sqlite3.Connection:
    """Drop the database file and re-init. Dev convenience; synthetic data only."""
    path = Path(db_path)
    if path.exists():
        path.unlink()
    conn = connect(path)
    init_db(conn)
    return conn


# ---------------------------------------------------------------------- sessions
def create_session(conn: sqlite3.Connection, language: str = "en-US") -> str:
    """Insert a new session row and return its id."""
    session_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO sessions (id, started_at, status, language) VALUES (?, ?, ?, ?)",
        (session_id, _now_iso(), "active", language),
    )
    conn.commit()
    return session_id


def get_session(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()


def add_message(
    conn: sqlite3.Connection,
    session_id: str,
    role: str,
    content: str,
    model: str | None = None,
) -> int:
    """Append a message; returns its autoincrement id."""
    cur = conn.execute(
        "INSERT INTO messages (session_id, role, content, model, ts) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, model, _now_iso()),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_messages(
    conn: sqlite3.Connection,
    session_id: str,
    before_id: int | None = None,
) -> list[sqlite3.Row]:
    """Return a session's messages in order; ``before_id`` excludes ids >= it.

    Used to rebuild conversation history for the agent each turn (stateless-per-turn).
    """
    if before_id is None:
        return conn.execute(
            "SELECT id, role, content, model FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return conn.execute(
        "SELECT id, role, content, model FROM messages "
        "WHERE session_id = ? AND id < ? ORDER BY id ASC",
        (session_id, before_id),
    ).fetchall()


def count_user_messages(conn: sqlite3.Connection, session_id: str) -> int:
    """Count patient turns so far (the turn number after persisting the current message)."""
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE session_id = ? AND role = 'user'",
        (session_id,),
    ).fetchone()
    return int(row["n"])


# ------------------------------------------------------------------ intake state
def load_intake_state(conn: sqlite3.Connection, session_id: str) -> IntakeState:
    """Reconstruct :class:`IntakeState` from SQLite (latest-wins per slot)."""
    row = get_session(conn, session_id)
    if row is None:
        raise ValueError(f"unknown session: {session_id}")

    signals = Signals(**json.loads(row["signals_json"])) if row["signals_json"] else Signals()

    slots: dict[str, SlotValue] = {}
    # ORDER BY id ASC => later writes overwrite earlier ones (latest-wins).
    for r in conn.execute(
        "SELECT slot, value, confidence, source_msg_id, updated_at "
        "FROM intake_state WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    ):
        slots[r["slot"]] = SlotValue(
            value=r["value"] if r["value"] is not None else "",
            confidence=Confidence(r["confidence"]) if r["confidence"] else Confidence.unknown,
            source_msg_id=r["source_msg_id"],
            updated_at=r["updated_at"],
        )

    floor = TriageBand(row["triage_floor"]) if row["triage_floor"] else TriageBand.self_care
    return IntakeState(
        session_id=session_id,
        slots=slots,
        triage_floor=floor,
        floor_pinned=bool(row["floor_pinned"]),
        signals=signals,
        status=row["status"] or "active",
    )


def save_intake_state(conn: sqlite3.Connection, state: IntakeState) -> None:
    """Persist state: scalar fields update ``sessions``; each slot appends an audit row."""
    conn.execute(
        "UPDATE sessions SET status = ?, triage_floor = ?, floor_pinned = ?, signals_json = ? "
        "WHERE id = ?",
        (
            state.status,
            state.triage_floor.value,
            int(state.floor_pinned),
            json.dumps(state.signals.model_dump()),
            state.session_id,
        ),
    )
    now = _now_iso()
    for slot, sv in state.slots.items():
        conn.execute(
            "INSERT INTO intake_state "
            "(session_id, slot, value, confidence, source_msg_id, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                state.session_id,
                slot,
                sv.value,
                sv.confidence.value if sv.confidence else None,
                sv.source_msg_id,
                sv.updated_at or now,
            ),
        )
    conn.commit()


# ------------------------------------------------------------------- summaries
def save_summary(
    conn: sqlite3.Connection,
    session_id: str,
    soap_json: str,
    version: str,
    ts: str | None = None,
) -> int:
    """Insert one ``summaries`` row (final SOAP); returns its id."""
    cur = conn.execute(
        "INSERT INTO summaries (session_id, soap_json, version, ts) VALUES (?, ?, ?, ?)",
        (session_id, soap_json, version, ts or _now_iso()),
    )
    conn.commit()
    return int(cur.lastrowid)


def finalize_session(
    conn: sqlite3.Connection,
    session_id: str,
    triage_band: str,
    completed_at: str | None = None,
) -> None:
    """Mark a session completed: set its final ``triage_band`` and ``completed_at``.

    The monotonic ``triage_floor`` and ``status`` are written by :func:`save_intake_state`;
    this sets only the two completion-specific columns.
    """
    conn.execute(
        "UPDATE sessions SET triage_band = ?, completed_at = ? WHERE id = ?",
        (triage_band, completed_at or _now_iso(), session_id),
    )
    conn.commit()


# --------------------------------------------------------------- observability
def log_tool_call(conn: sqlite3.Connection, trace: ToolCallTrace) -> int:
    """Insert one ``tool_calls`` audit row; returns its id."""
    cur = conn.execute(
        "INSERT INTO tool_calls ("
        "session_id, turn, tool, args_json, result_json, input_tokens, output_tokens, "
        "cache_read_tokens, cache_creation_tokens, latency_ms, model, cost_usd, "
        "prompt_version, rules_version, ts"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            trace.session_id,
            trace.turn,
            trace.tool,
            trace.args_json,
            trace.result_json,
            trace.input_tokens,
            trace.output_tokens,
            trace.cache_read_tokens,
            trace.cache_creation_tokens,
            trace.latency_ms,
            trace.model,
            trace.cost_usd,
            trace.prompt_version,
            trace.rules_version,
            trace.ts or _now_iso(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def log_safety_event(
    conn: sqlite3.Connection,
    session_id: str,
    level: str,
    source: str,
    matched_rules: list[str],
    rules_version: str,
    msg_id: str | None = None,
    ts: str | None = None,
) -> int:
    """Insert one ``safety_events`` audit row; returns its id."""
    cur = conn.execute(
        "INSERT INTO safety_events ("
        "session_id, level, source, matched_rules_json, rules_version, msg_id, ts"
        ") VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            session_id,
            level,
            source,
            json.dumps(matched_rules),
            rules_version,
            msg_id,
            ts or _now_iso(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)

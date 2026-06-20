"""Read trace data the observability layer reports over — the ``tool_calls`` table and the
eval ``runs/`` JSONL (Split 09).

The dashboard and cost report **read existing data**; they never call a model or stand up a
service. The two sources are:

* ``tool_calls`` rows in any SQLite DB (a live session's ``data/scribeintake.db``) — the rich
  source: every token bucket incl. ``cache_read_tokens`` (the caching proof) is persisted here.
* ``eval/runs/<ts>/*.jsonl`` — one :class:`~eval.models.ScenarioRun` per line; carries per-turn
  cost/latency for the fleet trend (but not the cache buckets, which only live in the DB).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TraceRow:
    """One ``tool_calls`` row, normalised for the cost/latency math.

    ``model`` is ``None`` for local (``$0``) tool rows; pricing helpers skip those.
    """

    session_id: str
    turn: int
    tool: str
    model: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    latency_ms: int
    cost_usd: float
    ts: str | None = None

    @property
    def is_model_call(self) -> bool:
        return bool(self.model)

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turn": self.turn,
            "tool": self.tool,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "ts": self.ts,
        }


def _row_to_trace(r: sqlite3.Row) -> TraceRow:
    return TraceRow(
        session_id=r["session_id"],
        turn=int(r["turn"] or 0),
        tool=r["tool"],
        model=r["model"],
        input_tokens=int(r["input_tokens"] or 0),
        output_tokens=int(r["output_tokens"] or 0),
        cache_read_tokens=int(r["cache_read_tokens"] or 0),
        cache_creation_tokens=int(r["cache_creation_tokens"] or 0),
        latency_ms=int(r["latency_ms"] or 0),
        cost_usd=float(r["cost_usd"] or 0.0),
        ts=r["ts"],
    )


def read_tool_calls(conn: sqlite3.Connection, session_id: str | None = None) -> list[TraceRow]:
    """Return ``tool_calls`` rows (optionally one session), oldest first."""
    if session_id is None:
        rows = conn.execute("SELECT * FROM tool_calls ORDER BY id ASC").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM tool_calls WHERE session_id = ? ORDER BY id ASC", (session_id,)
        ).fetchall()
    return [_row_to_trace(r) for r in rows]


def session_ids(conn: sqlite3.Connection) -> list[str]:
    """Distinct session ids that have at least one trace row, oldest first."""
    rows = conn.execute(
        "SELECT session_id, MIN(id) AS first FROM tool_calls GROUP BY session_id ORDER BY first"
    ).fetchall()
    return [r["session_id"] for r in rows]


def iter_run_records(runs_dir: str | Path) -> Iterator[dict]:
    """Yield raw :class:`ScenarioRun` dicts from every ``*.jsonl`` under ``runs_dir`` (recursive).

    Robust to a missing directory and to blank lines; the latest timestamped subdir is **not**
    privileged — all runs found are yielded (the caller decides how to aggregate).
    """
    root = Path(runs_dir)
    if not root.exists():
        return
    for path in sorted(root.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def latest_runs_dir(runs_dir: str | Path) -> Path | None:
    """The most recent timestamped subdir under ``runs_dir`` (lexicographic == chronological)."""
    root = Path(runs_dir)
    if not root.exists():
        return None
    subdirs = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
    return subdirs[0] if subdirs else None

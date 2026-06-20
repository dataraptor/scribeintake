"""Request-scoped wiring: DB connections, CORS config, settings (Split 10, spec §6).

The app is **stateless per request**: every handler opens a fresh SQLite connection (bound to
the app's configured DB path), runs, and closes it — there is no module-level session cache,
which keeps the API consistent with the eval harness's parallel-safe isolation.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from scribeintake import db
from scribeintake.config import settings


def default_db_path() -> Path:
    """The SQLite path the live server uses (the core settings default under ``data/``)."""
    return settings.DB_PATH


def open_conn(db_path: str | Path) -> sqlite3.Connection:
    """Open a fresh connection to ``db_path`` (per-request; the caller must close it)."""
    return db.connect(db_path)


def cors_origins() -> list[str]:
    """Allowed CORS origins for the frontend.

    Configurable via the ``API_CORS_ORIGINS`` env var (comma-separated); defaults to ``*`` for the
    single-instance demo. A production deployment should pin this to the frontend's exact origin.
    """
    raw = os.environ.get("API_CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]

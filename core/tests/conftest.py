"""Shared pytest fixtures + test-helper path setup for the core suite.

``--import-mode=importlib`` (root pyproject) doesn't put the test directory on ``sys.path``,
so add it here once to make the sibling ``fakes`` helper importable from any test module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from scribeintake import db  # noqa: E402  (after sys.path tweak)


@pytest.fixture
def conn(tmp_path):
    """A fresh, isolated SQLite database with all tables created."""
    connection = db.reset_db(tmp_path / "test.db")
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def session(conn):
    """A new session id in the fixture database."""
    return db.create_session(conn)

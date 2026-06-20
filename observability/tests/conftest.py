"""Shared fixtures for the observability suite.

``--import-mode=importlib`` (root pyproject) doesn't put the core test helpers on ``sys.path``;
the observability tests only need a fresh DB, which the ``scribeintake.db`` reset helper provides.
"""

from __future__ import annotations

import pytest

from scribeintake import db


@pytest.fixture
def conn(tmp_path):
    """A fresh, isolated SQLite database with all tables created."""
    connection = db.reset_db(tmp_path / "obs.db")
    try:
        yield connection
    finally:
        connection.close()

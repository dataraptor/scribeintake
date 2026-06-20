"""Test-helper path setup for the eval suite.

``--import-mode=importlib`` (root pyproject) doesn't put test directories on ``sys.path``, so
expose the **core** test helpers (``fakes``) here — the harness tests reuse the same scripted
``FakeLLMClient`` / ``FakeStructuredClient`` the orchestrator tests use, so no key is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scribeintake/eval/tests/conftest.py -> parents[2] == repo root -> core/tests holds `fakes`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "tests"))

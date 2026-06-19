"""Live end-to-end test against the real model (Azure OpenAI GPT-5.5).

Marked ``live`` — excluded from the per-commit gate (``-m "not live"``). Runs on demand:

    python -m pytest core/tests/test_agent_live.py -m live -v

Skips cleanly if Azure credentials aren't configured. Kept to a single short turn to bound
cost. NOTE: the spec pins Claude; this environment ships an Azure GPT-5.5 key instead, so the
live agent runs on GPT-5.5 (a reasoning model that, like Opus, rejects sampling knobs).
"""

from __future__ import annotations

import pytest

from scribeintake import db
from scribeintake.config import MAX_AGENT_STEPS, settings
from scribeintake.orchestrator import run_turn

pytestmark = pytest.mark.live


@pytest.fixture
def conn(tmp_path):
    connection = db.reset_db(tmp_path / "live.db")
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture(autouse=True)
def _require_azure():
    if not (settings.azure_openai_endpoint and settings.azure_openai_api_key):
        pytest.skip("Azure OpenAI credentials not configured (.env)")


def test_live_benign_intake_turn(conn):
    session = db.create_session(conn)
    turn = run_turn(
        session,
        "Hi, I've had a sore throat and a mild cough for about two days.",
        conn=conn,
    )

    # A real assistant question came back and the conversation continues.
    assert turn.assistant_text.strip()
    assert turn.status in {"active", "ready_to_summarize"}
    assert turn.level.value in {"CLEAR", "URGENT"}  # benign opener, not an emergency
    assert turn.model == settings.ACTIVE_INTAKE_MODEL

    # The agent recorded structured intake from the message.
    assert "record_intake" in turn.tools_used

    # Bounded loop: at most MAX_AGENT_STEPS model calls.
    model_steps = [t for t in turn.traces if t.tool == "agent_step"]
    assert 1 <= len(model_steps) <= MAX_AGENT_STEPS

    # Usage and cost were recorded for the model calls.
    assert all(t.input_tokens > 0 for t in model_steps)
    total_cost = sum(t.cost_usd for t in model_steps)
    assert total_cost > 0

    print(
        f"\n[live] model={turn.model} steps={len(model_steps)} "
        f"tools={turn.tools_used} cost=${total_cost:.6f} "
        f"q={turn.assistant_text[:80]!r}"
    )

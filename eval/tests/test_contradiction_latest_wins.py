"""Contradiction / correction resolves latest-wins with an audit trail (Split 13 §3.4).

Driven through the **harness** with a mocked agent (no API key): a slot is recorded, then a
later turn corrects it. The reconstructed state must reflect the **latest** value (latest-wins),
and the append-only ``intake_state`` table must retain **both** writes (the audit trail) — so a
correction is honored without losing the history of what was said.
"""

from __future__ import annotations

import sqlite3

from fakes import FakeLLMClient, text_response, tool_response

from eval.harness import run_scenario
from eval.scenario import Expect, Scenario, ScenarioCategory
from scribeintake import db
from scribeintake.agent import AgentLoop
from scribeintake.models import EscalationLevel
from scribeintake.tools import default_registry


def _record(updates):
    return tool_response(
        [("record_intake", {"updates": [{"slot": s, "value": v} for s, v in updates]})]
    )


def _correcting_agent() -> AgentLoop:
    # Per turn: one record call, then a plain-text question ends the turn.
    return AgentLoop(
        FakeLLMClient(
            [
                _record([("chief_complaint", "cough"), ("hpi.onset", "today")]),
                text_response("How long has the cough been going on?"),
                _record([("hpi.onset", "three days ago")]),
                text_response("Anything else you'd like to add?"),
            ]
        ),
        default_registry(),
    )


def _correction_scenario() -> Scenario:
    return Scenario(
        id="contradiction_latest_wins_x",
        category=ScenarioCategory.routine,
        turns=[
            "I've had a cough since today",
            "actually, correction — it started three days ago",
        ],
        expect=Expect(escalation=EscalationLevel.CLEAR),
        provenance="synthetic",
    )


def test_correction_is_latest_wins_with_a_full_audit_trail(tmp_path):
    db_file = str(tmp_path / "contradiction.db")
    run = run_scenario(
        _correction_scenario(),
        seed_label="run-1",
        db_path=db_file,  # caller-owned ⇒ the harness leaves it for us to inspect
        agent=_correcting_agent(),
    )
    assert run.n_turns_run == 2
    assert run.intake_halted is False

    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    try:
        session_id = conn.execute("SELECT id FROM sessions LIMIT 1").fetchone()["id"]

        # Audit trail: both writes to the corrected slot survive, in order (append-only).
        onset_rows = conn.execute(
            "SELECT value FROM intake_state WHERE session_id = ? AND slot = 'hpi.onset' "
            "ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        assert [r["value"] for r in onset_rows] == ["today", "three days ago"]

        # Latest-wins: the reconstructed state reflects the correction.
        state = db.load_intake_state(conn, session_id)
        assert state.slots["hpi.onset"].value == "three days ago"
        assert state.slots["chief_complaint"].value == "cough"
    finally:
        conn.close()

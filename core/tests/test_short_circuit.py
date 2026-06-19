"""The headline behavioral guarantee: on a gate EMERGENCY the agent NEVER runs.

These assert the call-count (the proof), not just the output text — the safety guarantee is
that step 3 (code) short-circuits *upstream* of any model call.
"""

from __future__ import annotations

from unittest.mock import Mock

from scribeintake import db
from scribeintake.models import EscalationLevel, EscalationSource
from scribeintake.orchestrator import run_turn


def _safety_events(conn, session_id):
    return conn.execute(
        "SELECT level, source FROM safety_events WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()


def test_gate_emergency_short_circuits_agent_never_runs(conn, session):
    agent = Mock()  # if the orchestrator touches it, call_count proves the bug
    turn = run_turn(
        session,
        "I have crushing chest pain spreading to my left arm and I'm sweating.",
        conn=conn,
        agent=agent,
    )

    # The agent was never invoked.
    assert agent.run_turn.call_count == 0
    # Emergency template emitted (code, not a model call).
    assert turn.level is EscalationLevel.EMERGENCY
    assert turn.source is EscalationSource.gate
    assert turn.template is not None
    assert turn.template["kind"] == "emergency"
    assert "911" in turn.assistant_text
    # Session halted and persisted.
    assert turn.status == "halted"
    assert db.get_session(conn, session)["status"] == "halted"
    # safety_event logged with source=gate at EMERGENCY.
    events = _safety_events(conn, session)
    assert ("EMERGENCY", "gate") in [(e["level"], e["source"]) for e in events]


def test_crisis_message_routes_to_crisis_template(conn, session):
    agent = Mock()
    turn = run_turn(
        session,
        "I've been thinking about suicide and I want to end my life.",
        conn=conn,
        agent=agent,
    )
    assert agent.run_turn.call_count == 0
    assert turn.level is EscalationLevel.EMERGENCY
    assert turn.crisis is True
    assert turn.template["kind"] == "crisis"
    assert "988" in turn.assistant_text
    assert turn.status == "halted"


def test_no_assistant_model_message_persisted_on_short_circuit(conn, session):
    agent = Mock()
    run_turn(session, "chest pain radiating to my jaw, can't breathe", conn=conn, agent=agent)
    # The persisted assistant message has no model attribution (template, not a model call).
    rows = db.get_messages(conn, session)
    assistant = [r for r in rows if r["role"] == "assistant"]
    assert len(assistant) == 1
    assert assistant[0]["model"] is None
    # And no tool_calls trace rows (agent never ran).
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM tool_calls WHERE session_id = ?", (session,)
    ).fetchone()["n"]
    assert n == 0

"""Deterministic orchestration tests (LLM mocked, no key/network).

Covers the non-emergency paths: URGENT pin+continue, monotonic floor, the orchestrator-owned
completion check, the agent's independent escalation net (escalate-only), refusal handling,
trace/cost accounting, and statelessness across turns.
"""

from __future__ import annotations

from fakes import FakeLLMClient, refusal_response, text_response, tool_response

from scribeintake import db
from scribeintake.agent import REFUSAL_REPLY, AgentLoop
from scribeintake.models import EscalationLevel, EscalationSource, TriageBand
from scribeintake.orchestrator import run_turn
from scribeintake.tools import default_registry

CLEAR_MSG = "I've had a mild sore throat for a couple of days."
URGENT_MSG = "I have severe abdominal pain that is really bad."


def agent_with(*responses) -> AgentLoop:
    """A real AgentLoop wired to a scripted fake client (exercises real tool dispatch)."""
    return AgentLoop(FakeLLMClient(list(responses)), default_registry())


def _record(updates: list[tuple[str, str]]):
    return tool_response(
        [("record_intake", {"updates": [{"slot": s, "value": v} for s, v in updates]})]
    )


def seed_then_ask(slot: str, value: str, question: str = "Since when?") -> AgentLoop:
    """An agent that records one slot then asks a question."""
    return agent_with(_record([(slot, value)]), text_response(question))


def _safety_events(conn, session_id):
    return [
        (e["level"], e["source"])
        for e in conn.execute(
            "SELECT level, source FROM safety_events WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    ]


# --------------------------------------------------------------------- URGENT path
def test_urgent_gate_pins_floor_and_continues(conn, session):
    agent = agent_with(
        _record([("chief_complaint", "abdominal pain")]),
        text_response("Where is the pain?"),
    )
    turn = run_turn(session, URGENT_MSG, conn=conn, agent=agent)

    assert turn.level is EscalationLevel.URGENT
    assert turn.source is EscalationSource.gate
    assert turn.triage_floor is TriageBand.gp_urgent
    assert turn.template is not None and turn.template["kind"] == "urgent"
    assert turn.status == "active"  # urgent continues
    assert turn.assistant_text == "Where is the pain?"  # the agent ran and asked


def test_floor_is_monotonic_across_turns(conn, session):
    # Turn 1: URGENT pins gp_urgent.
    run_turn(session, URGENT_MSG, conn=conn, agent=agent_with(text_response("Tell me more.")))
    assert db.load_intake_state(conn, session).triage_floor is TriageBand.gp_urgent
    # Turn 2: a CLEAR message must NOT lower the floor.
    run_turn(session, CLEAR_MSG, conn=conn, agent=agent_with(text_response("And anything else?")))
    state = db.load_intake_state(conn, session)
    assert state.triage_floor is TriageBand.gp_urgent
    assert state.floor_pinned is True


# ---------------------------------------------- completion check (orchestrator-owned)
def test_completion_check_is_orchestrator_owned_not_the_agent(conn, session):
    # Turn 1 fills two slots; not yet complete.
    t1 = run_turn(
        session,
        CLEAR_MSG,
        conn=conn,
        agent=agent_with(
            _record([("chief_complaint", "sore throat"), ("onset", "2 days ago")]),
            text_response("How long has it lasted?"),
        ),
    )
    assert t1.status == "active"
    assert db.load_intake_state(conn, session).slots["chief_complaint"].value == "sore throat"

    # Turn 2 fills the remaining required slots; the *engine* flips ready_to_summarize,
    # even though the agent's text never claims to be done.
    t2 = run_turn(
        session,
        "Here are the rest of the details.",
        conn=conn,
        agent=agent_with(
            _record(
                [
                    ("duration", "2 days"),
                    ("severity", "mild"),
                    ("associated_symptoms", "none"),
                    ("medications", "none"),
                    ("allergies", "none"),
                ]
            ),
            text_response("Thanks — I'll prepare a summary for your clinician."),
        ),
    )
    assert t2.open_slots == []
    assert t2.status == "ready_to_summarize"


# ------------------------------------------ agent independent escalation net (§8)
def test_agent_emergency_is_treated_like_gate_emergency(conn, session):
    agent = agent_with(
        tool_response([("assess_escalation", {"level": "EMERGENCY", "rationale": "red flag"})]),
        text_response("this pending question must be discarded"),
    )
    turn = run_turn(session, "I just feel a bit off today.", conn=conn, agent=agent)

    assert turn.level is EscalationLevel.EMERGENCY
    assert turn.source is EscalationSource.agent
    assert turn.status == "halted"
    assert turn.template["kind"] == "emergency"
    assert "911" in turn.assistant_text
    # The pending agent question was discarded (template stands).
    assert "discarded" not in turn.assistant_text
    # safety_event logged with source=agent.
    assert ("EMERGENCY", "agent") in _safety_events(conn, session)


def test_agent_urgent_raises_floor_when_gate_is_clear(conn, session):
    agent = agent_with(
        tool_response([("assess_escalation", {"level": "URGENT", "rationale": "same-day"})]),
        text_response("Can you describe it more?"),
    )
    turn = run_turn(session, CLEAR_MSG, conn=conn, agent=agent)
    assert turn.level is EscalationLevel.URGENT
    assert turn.source is EscalationSource.agent
    assert turn.triage_floor is TriageBand.gp_urgent
    assert turn.status == "active"
    assert ("URGENT", "agent") in _safety_events(conn, session)


def test_agent_cannot_deescalate_below_gate_floor(conn, session):
    # Gate says URGENT; the agent asserting CLEAR must not lower the floor.
    agent = agent_with(
        tool_response([("assess_escalation", {"level": "CLEAR", "rationale": "looks fine"})]),
        text_response("Where exactly does it hurt?"),
    )
    turn = run_turn(session, URGENT_MSG, conn=conn, agent=agent)
    assert turn.level is EscalationLevel.URGENT
    assert turn.source is EscalationSource.gate
    assert turn.triage_floor is TriageBand.gp_urgent


# --------------------------------------------------------------- refusal handling
def test_refusal_is_handled_without_crashing_and_preserves_state(conn, session):
    # Seed a slot on turn 1.
    run_turn(session, CLEAR_MSG, conn=conn, agent=seed_then_ask("chief_complaint", "sore throat"))
    # Turn 2 the model refuses; the turn must not crash and state must survive.
    refuser = agent_with(refusal_response())
    turn = run_turn(session, "can you diagnose me?", conn=conn, agent=refuser)
    assert turn.assistant_text == REFUSAL_REPLY
    assert turn.status == "active"
    # The previously collected slot is still there.
    assert db.load_intake_state(conn, session).slots["chief_complaint"].value == "sore throat"


# --------------------------------------------------------------- trace + cost rows
def test_trace_rows_have_cost_for_model_and_zero_for_local(conn, session):
    run_turn(session, CLEAR_MSG, conn=conn, agent=seed_then_ask("chief_complaint", "sore throat"))
    rows = conn.execute(
        "SELECT tool, model, cost_usd FROM tool_calls WHERE session_id = ? ORDER BY id",
        (session,),
    ).fetchall()
    model_rows = [r for r in rows if r["tool"] == "agent_step"]
    local_rows = [r for r in rows if r["tool"] == "record_intake"]
    assert len(model_rows) == 2  # two model calls (tool turn + question turn)
    assert len(local_rows) == 1
    assert all(r["cost_usd"] > 0 and r["model"] for r in model_rows)
    assert all(r["cost_usd"] == 0 and r["model"] is None for r in local_rows)


# ------------------------------------------------------------------- statelessness
def test_orchestrator_is_stateless_across_turns(conn, session):
    # Each turn uses a *fresh* agent; continuity can only come from SQLite, not memory.
    run_turn(
        session,
        CLEAR_MSG,
        conn=conn,
        agent=agent_with(_record([("chief_complaint", "sore throat")]), text_response("q1")),
    )
    run_turn(
        session,
        "It started yesterday.",
        conn=conn,
        agent=agent_with(_record([("onset", "yesterday")]), text_response("q2")),
    )
    state = db.load_intake_state(conn, session)
    assert state.slots["chief_complaint"].value == "sore throat"
    assert state.slots["onset"].value == "yesterday"
    # Two patient turns recorded.
    assert db.count_user_messages(conn, session) == 2

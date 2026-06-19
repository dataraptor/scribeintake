"""End-to-end tests for ``run_gate`` — the canonical mockup scenarios + templates.

These mirror the four reference conversations in the ``.dc.html`` mockup so the Python
gate is provably at least as safe as the proven JS floor, and assert that each verdict
selects the right deterministic template (emergency / crisis / urgent / none).
"""

import json

from scribeintake.db import connect, create_session, init_db
from scribeintake.models import EscalationLevel, TriageBand
from scribeintake.safety import extract, run_gate


def test_acs_two_turns_emergency():
    """'chest tightness + left arm hurts' then 'sweating' -> EMERGENCY (acs_chest_pain)."""
    r1 = run_gate("I've had chest tightness and my left arm hurts")
    assert r1.verdict.level is EscalationLevel.EMERGENCY
    assert "acs_chest_pain" in r1.verdict.matched_rules
    # Even carrying prior signals forward, a second turn stays EMERGENCY.
    r2 = run_gate("yeah I'm sweating", r1.signals, r1.floor)
    assert r2.verdict.level is EscalationLevel.EMERGENCY
    assert r2.floor is TriageBand.ER
    assert r2.template["kind"] == "emergency"


def test_stroke_fast_emergency():
    r = run_gate("my face feels droopy on one side and my arm went weak")
    assert r.verdict.level is EscalationLevel.EMERGENCY
    assert "stroke_fast" in r.verdict.matched_rules
    assert r.template["kind"] == "emergency"


def test_hypertensive_urgency_not_emergency():
    """'186 over 122' then 'no chest pain or headache' -> URGENT (htn_urgency)."""
    prior = extract("my home monitor said 186 over 122")
    r = run_gate("no chest pain or headache", prior)
    assert r.verdict.level is EscalationLevel.URGENT
    assert "htn_urgency" in r.verdict.matched_rules
    assert r.floor is TriageBand.gp_urgent
    assert r.template["kind"] == "urgent"
    assert r.template["floor"] == "gp_urgent"


def test_benign_musculoskeletal_is_clear():
    """The false-alarm probe: chest-wall soreness must NOT escalate."""
    r = run_gate(
        "my chest is a bit sore after I helped move boxes; hurts when I press, "
        "no sweating or breathlessness"
    )
    assert r.verdict.level is EscalationLevel.CLEAR
    assert r.verdict.matched_rules == []
    assert r.template is None
    assert r.floor is TriageBand.self_care


def test_crisis_routes_to_crisis_template():
    r = run_gate("honestly I don't want to be here anymore, I want to end it all")
    assert r.verdict.level is EscalationLevel.EMERGENCY
    assert r.verdict.crisis is True
    assert r.template["kind"] == "crisis"
    body = r.template["body"]
    assert "988" in body
    assert "911" in body
    assert "educational demo" in body.lower()
    # Two tel: actions (988 + 911).
    hrefs = [a["href"] for a in r.template["actions"]]
    assert "tel:988" in hrefs
    assert "tel:911" in hrefs


def test_emergency_template_has_911_action():
    r = run_gate("I can't breathe")
    assert r.template["kind"] == "emergency"
    assert r.template["actions"][0]["href"] == "tel:911"


def test_gate_logs_safety_event_when_db_provided(tmp_path):
    conn = connect(tmp_path / "gate.db")
    init_db(conn)
    session_id = create_session(conn)

    run_gate("I can't breathe", conn=conn, session_id=session_id, msg_id="m1")

    rows = conn.execute(
        "SELECT level, source, matched_rules_json FROM safety_events WHERE session_id = ?",
        (session_id,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["level"] == "EMERGENCY"
    assert rows[0]["source"] == "gate"
    assert "respiratory_distress" in json.loads(rows[0]["matched_rules_json"])
    conn.close()


def test_gate_works_without_db():
    """The gate is unit-testable with no connection (DB is optional)."""
    r = run_gate("I have a stiff neck and a fever")
    assert r.verdict.level is EscalationLevel.EMERGENCY
    assert "meningitis_signs" in r.verdict.matched_rules
    assert r.failed_safe is False


def test_result_as_dict_is_json_serializable():
    r = run_gate("I can't breathe")
    blob = json.dumps(r.as_dict())
    assert "EMERGENCY" in blob

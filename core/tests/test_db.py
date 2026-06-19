"""Deterministic tests for the SQLite access layer (temp DB, no network)."""

from scribeintake.db import (
    add_message,
    connect,
    create_session,
    get_session,
    init_db,
    load_intake_state,
    log_safety_event,
    log_tool_call,
    save_intake_state,
)
from scribeintake.models import (
    Confidence,
    SlotValue,
    ToolCallTrace,
    TriageBand,
)

_EXPECTED_TABLES = {
    "sessions",
    "messages",
    "intake_state",
    "summaries",
    "tool_calls",
    "safety_events",
    "kb_chunks",
}


def _fresh_db(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_db(conn)
    return conn


def test_init_db_creates_all_seven_tables(tmp_path):
    conn = _fresh_db(tmp_path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert _EXPECTED_TABLES <= names


def test_session_message_intake_round_trip(tmp_path):
    conn = _fresh_db(tmp_path)
    session_id = create_session(conn)

    msg_id = add_message(conn, session_id, "user", "I have chest tightness")
    assert msg_id > 0

    state = load_intake_state(conn, session_id)
    assert state.session_id == session_id
    assert state.status == "active"
    assert state.triage_floor is TriageBand.self_care
    assert state.slots == {}


def test_save_intake_state_is_latest_wins_with_audit(tmp_path):
    conn = _fresh_db(tmp_path)
    session_id = create_session(conn)

    state = load_intake_state(conn, session_id)
    state.slots["chief_complaint"] = SlotValue(value="headache", confidence=Confidence.high)
    save_intake_state(conn, state)

    state.slots["chief_complaint"] = SlotValue(value="chest pain", confidence=Confidence.high)
    save_intake_state(conn, state)

    reloaded = load_intake_state(conn, session_id)
    assert reloaded.slots["chief_complaint"].value == "chest pain"

    # Both writes retained as audit rows.
    count = conn.execute(
        "SELECT COUNT(*) FROM intake_state WHERE session_id = ? AND slot = 'chief_complaint'",
        (session_id,),
    ).fetchone()[0]
    assert count == 2


def test_save_intake_state_persists_floor_and_signals(tmp_path):
    conn = _fresh_db(tmp_path)
    session_id = create_session(conn)

    state = load_intake_state(conn, session_id)
    state.triage_floor = TriageBand.gp_urgent
    state.floor_pinned = True
    state.status = "halted"
    state.signals.chest_pain = True
    state.signals.sbp = 186
    save_intake_state(conn, state)

    reloaded = load_intake_state(conn, session_id)
    assert reloaded.triage_floor is TriageBand.gp_urgent
    assert reloaded.floor_pinned is True
    assert reloaded.status == "halted"
    assert reloaded.signals.chest_pain is True
    assert reloaded.signals.sbp == 186


def test_unknown_session_raises(tmp_path):
    conn = _fresh_db(tmp_path)
    try:
        load_intake_state(conn, "does-not-exist")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown session")


def test_log_tool_call_and_safety_event(tmp_path):
    conn = _fresh_db(tmp_path)
    session_id = create_session(conn)

    trace = ToolCallTrace(
        session_id=session_id,
        turn=1,
        tool="retrieve_guideline",
        model="claude-sonnet-4-6",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
    )
    row_id = log_tool_call(conn, trace)
    assert row_id > 0

    persisted = conn.execute(
        "SELECT tool, prompt_version, rules_version FROM tool_calls WHERE id = ?",
        (row_id,),
    ).fetchone()
    assert persisted["tool"] == "retrieve_guideline"
    assert persisted["prompt_version"] == "v1"
    assert persisted["rules_version"] == "v1"

    event_id = log_safety_event(
        conn,
        session_id,
        level="EMERGENCY",
        source="gate",
        matched_rules=["acs_chest_pain"],
        rules_version="v1",
        msg_id="m1",
    )
    assert event_id > 0
    event = conn.execute(
        "SELECT level, source, matched_rules_json FROM safety_events WHERE id = ?",
        (event_id,),
    ).fetchone()
    assert event["level"] == "EMERGENCY"
    assert event["source"] == "gate"
    assert "acs_chest_pain" in event["matched_rules_json"]


def test_get_session_returns_row(tmp_path):
    conn = _fresh_db(tmp_path)
    session_id = create_session(conn, language="en-US")
    row = get_session(conn, session_id)
    assert row is not None
    assert row["id"] == session_id
    assert row["language"] == "en-US"

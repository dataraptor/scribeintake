"""Endpoint tests with the orchestrator MOCKED (deterministic, no API key)."""

from __future__ import annotations

import json

import pytest

from api.tests.conftest import (
    make_clear_turn,
    make_completed_turn,
    make_gate_emergency_turn,
    sample_soap,
)
from scribeintake import db

JSON = {"accept": "application/json"}


# --------------------------------------------------------------------------------- /health
def test_health(client):
    out = client.get("/health").json()
    assert out["status"] == "ok"
    assert out["version"]
    assert "intake" in out["models"] and "summary" in out["models"]


# -------------------------------------------------------------------------------- /session
def test_create_session_returns_id_and_disclaimer(client):
    resp = client.post("/session")
    assert resp.status_code == 200
    out = resp.json()
    assert out["sessionId"]
    assert out["disclaimer"]


# ------------------------------------------------------------------------ /message (mocked)
def test_message_non_streaming_returns_turn(client, session_id, monkeypatch):
    turn = make_clear_turn(session_id)
    monkeypatch.setattr("api.main.run_turn", lambda sid, text, *, conn: turn)

    url = f"/session/{session_id}/message"
    resp = client.post(url, json={"text": "my head hurts"}, headers=JSON)
    assert resp.status_code == 200
    out = resp.json()
    assert out["content"] == turn.assistant_text
    assert out["level"] == "CLEAR"
    assert out["strip"]["ruleId"] == "no rule matched"
    assert out["emergency"] is None


def test_message_emergency_returns_halted_with_payload(client, session_id, monkeypatch):
    turn = make_gate_emergency_turn(session_id)
    monkeypatch.setattr("api.main.run_turn", lambda sid, text, *, conn: turn)

    resp = client.post(
        f"/session/{session_id}/message",
        json={"text": "crushing chest pain to my arm, sweating"},
        headers=JSON,
    )
    out = resp.json()
    assert out["status"] == "halted"
    assert out["level"] == "EMERGENCY"
    assert out["emergency"]["actions"][0]["href"] == "tel:911"


def test_message_unknown_session_is_404_not_500(client, monkeypatch):
    # run_turn should never be reached for an unknown session
    monkeypatch.setattr(
        "api.main.run_turn",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("run_turn must not run")),
    )
    resp = client.post("/session/does-not-exist/message", json={"text": "hi"}, headers=JSON)
    assert resp.status_code == 404
    out = resp.json()
    assert out["error"] == "session_not_found"
    assert "does-not-exist" in out["detail"]


def test_message_model_error_returns_friendly_503(client, session_id, monkeypatch):
    def boom(sid, text, *, conn):
        raise RuntimeError("simulated upstream failure")

    monkeypatch.setattr("api.main.run_turn", boom)
    resp = client.post(f"/session/{session_id}/message", json={"text": "hi"}, headers=JSON)
    assert resp.status_code == 503
    out = resp.json()
    assert out["error"] == "upstream_unavailable"
    assert "saved" in out["detail"]  # friendly reconnect, never a blank 500


def test_message_preserves_state_on_error(client, session_id, monkeypatch):
    """The patient message is persisted before any model call, so state survives an error."""

    def boom(sid, text, *, conn):
        # mimic the orchestrator persisting the user message before the model step
        db.add_message(conn, sid, "user", text)
        raise RuntimeError("simulated failure after persist")

    monkeypatch.setattr("api.main.run_turn", boom)
    client.post(f"/session/{session_id}/message", json={"text": "remember me"}, headers=JSON)

    # the message is still in the DB (state preserved for a retry)
    conn = db.connect(client.app.state.db_path)
    try:
        msgs = db.get_messages(conn, session_id)
    finally:
        conn.close()
    assert any(m["content"] == "remember me" for m in msgs)


# ------------------------------------------------------------------------------- /summary
def test_summary_returns_soap(client, session_id):
    # persist a SOAP + finalize the session as the orchestrator would
    conn = db.connect(client.app.state.db_path)
    try:
        db.save_summary(conn, session_id, json.dumps(sample_soap()), "v1")
        db.finalize_session(conn, session_id, "gp_routine")
    finally:
        conn.close()

    out = client.get(f"/session/{session_id}/summary").json()
    assert out["band"] == "gp_routine"
    assert out["disclaimer"]
    assert out["red_flags_checked"] == 21
    assert any(o["cited"] for o in out["observations"])


def test_summary_404_with_reason_when_none(client, session_id):
    resp = client.get(f"/session/{session_id}/summary")
    assert resp.status_code == 404
    out = resp.json()
    assert out["error"] == "no_summary"
    assert "not complete" in out["detail"]


def test_summary_reason_for_halted_session(client, session_id):
    conn = db.connect(client.app.state.db_path)
    try:
        conn.execute("UPDATE sessions SET status = 'halted' WHERE id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()
    out = client.get(f"/session/{session_id}/summary").json()
    assert "safety referral" in out["detail"]


def test_summary_unknown_session_404(client):
    resp = client.get("/session/nope/summary")
    assert resp.status_code == 404
    assert resp.json()["error"] == "session_not_found"


# --------------------------------------------------------------------------------- /trace
def test_trace_returns_rows_and_totals(client, session_id):
    # write a couple of trace rows via the engine's logger
    turn = make_completed_turn(session_id)
    conn = db.connect(client.app.state.db_path)
    try:
        for tr in turn.traces:
            tr.session_id = session_id
            db.log_tool_call(conn, tr)
    finally:
        conn.close()

    out = client.get(f"/session/{session_id}/trace").json()
    assert out["sessionId"] == session_id
    assert out["rows"]
    assert out["totalCostUsd"] > 0
    assert out["traceCostLabel"].startswith("$")


def test_trace_empty_session_is_zeroed(client, session_id):
    out = client.get(f"/session/{session_id}/trace").json()
    assert out["rows"] == []
    assert out["totalCostUsd"] == 0.0


def test_trace_unknown_session_404(client):
    assert client.get("/session/nope/trace").status_code == 404


# ------------------------------------------------------------- statelessness (no session map)
def test_stateless_fresh_connection_per_request(client, monkeypatch):
    """Each request opens its own DB connection — there is no in-memory session map."""
    seen_conn_ids = []

    def capture(sid, text, *, conn):
        seen_conn_ids.append(id(conn))
        return make_clear_turn(sid)

    monkeypatch.setattr("api.main.run_turn", capture)
    sid = client.post("/session").json()["sessionId"]
    client.post(f"/session/{sid}/message", json={"text": "one"}, headers=JSON)
    client.post(f"/session/{sid}/message", json={"text": "two"}, headers=JSON)

    assert len(seen_conn_ids) == 2
    assert seen_conn_ids[0] != seen_conn_ids[1]  # distinct connections, not a shared cache
    assert not hasattr(client.app.state, "sessions")  # no module-level session store


@pytest.mark.parametrize("path", ["/session/x/summary", "/session/x/trace"])
def test_unknown_session_never_500(client, path):
    assert client.get(path).status_code == 404

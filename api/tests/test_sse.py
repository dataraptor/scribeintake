"""SSE framing tests (orchestrator mocked): token frames then a terminal turn event; friendly
error frame on a mid-stream failure (§3.2/§18)."""

from __future__ import annotations

import json

from api.tests.conftest import make_clear_turn, make_gate_emergency_turn

SSE = {"accept": "text/event-stream"}


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into ``[(event, data_dict), ...]``."""
    events: list[tuple[str, dict]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        event = "message"
        data = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = line[len("data:") :].strip()
        events.append((event, json.loads(data) if data else {}))
    return events


def test_stream_emits_tokens_then_terminal_turn(client, session_id, monkeypatch):
    turn = make_clear_turn(session_id, assistant_text="Can you tell me when this started?")
    monkeypatch.setattr("api.main.run_turn", lambda sid, text, *, conn, on_event=None: turn)

    url = f"/session/{session_id}/message"
    resp = client.post(url, json={"text": "my head hurts"}, headers=SSE)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(resp.text)
    kinds = [e for e, _ in events]
    assert "token" in kinds
    assert kinds[-1] == "turn", "the final event must be the full turn payload"

    # token frames reassemble the assistant text
    streamed = "".join(d["text"] for e, d in events if e == "token")
    assert streamed == turn.assistant_text

    # terminal turn carries the full payload (strip, level, status)
    _, final = events[-1]
    assert final["content"] == turn.assistant_text
    assert final["level"] == "CLEAR"
    assert final["strip"]["ruleId"] == "no rule matched"


def test_stream_emergency_turn_carries_sheet(client, session_id, monkeypatch):
    turn = make_gate_emergency_turn(session_id)
    monkeypatch.setattr("api.main.run_turn", lambda sid, text, *, conn, on_event=None: turn)

    resp = client.post(f"/session/{session_id}/message", json={"text": "chest pain"}, headers=SSE)
    events = parse_sse(resp.text)
    _, final = events[-1]
    assert final["status"] == "halted"
    assert final["emergency"]["actions"][0]["href"] == "tel:911"


def test_stream_error_emits_friendly_error_frame(client, session_id, monkeypatch):
    def boom(sid, text, *, conn, on_event=None):
        raise RuntimeError("simulated mid-stream failure")

    monkeypatch.setattr("api.main.run_turn", boom)
    resp = client.post(f"/session/{session_id}/message", json={"text": "hi"}, headers=SSE)
    assert resp.status_code == 200  # the stream opened; the error is a frame, not a dropped conn

    events = parse_sse(resp.text)
    assert events, "must not be a blank/dropped stream"
    kinds = [e for e, _ in events]
    assert "error" in kinds
    _, err = next((e, d) for e, d in events if e == "error")
    assert err["kind"] == "reconnect"
    assert "saved" in err["message"]


def test_default_accept_streams(client, session_id, monkeypatch):
    """No explicit Accept (TestClient sends */*) defaults to the SSE stream."""
    turn = make_clear_turn(session_id)
    monkeypatch.setattr("api.main.run_turn", lambda sid, text, *, conn, on_event=None: turn)
    resp = client.post(f"/session/{session_id}/message", json={"text": "hi"})
    assert resp.headers["content-type"].startswith("text/event-stream")


def test_stream_relays_progress_stage_frames_before_tokens(client, session_id, monkeypatch):
    """The orchestrator's ``on_event`` progress callbacks surface as ``stage`` frames, in order,
    before the token/turn frames — this is the live "what the agent is doing" view."""
    turn = make_clear_turn(session_id, assistant_text="When did it start?")

    def fake_run(sid, text, *, conn, on_event=None):
        # Emit the same shape the real orchestrator/agent emit.
        on_event({"stage": "gate", "label": "Screening for urgent red-flag symptoms"})
        on_event({"stage": "thinking", "label": "Thinking…"})
        on_event({"stage": "tool", "tool": "record_intake", "label": "Recording what you told me"})
        return turn

    monkeypatch.setattr("api.main.run_turn", fake_run)

    resp = client.post(f"/session/{session_id}/message", json={"text": "my head hurts"}, headers=SSE)
    assert resp.status_code == 200
    events = parse_sse(resp.text)
    kinds = [e for e, _ in events]

    # all three stage frames arrive, in order, ahead of the first token and the terminal turn
    assert kinds.count("stage") == 3
    first_token = kinds.index("token")
    assert all(i < first_token for i, k in enumerate(kinds) if k == "stage")
    stages = [d for e, d in events if e == "stage"]
    assert [s["stage"] for s in stages] == ["gate", "thinking", "tool"]
    assert stages[2]["tool"] == "record_intake"
    assert kinds[-1] == "turn"

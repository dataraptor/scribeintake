"""Live API smoke (@pytest.mark.live) — a real 1-2 turn intake over the TestClient.

Needs LLM credentials (AZURE_OPENAI_API_KEY / endpoint, from .env). Kept short to bound cost:
one benign turn (a single model question) + one danger turn (gate short-circuit, no model call).
"""

from __future__ import annotations

import json

import pytest

from api.tests.test_sse import parse_sse

SSE = {"accept": "text/event-stream"}
JSON = {"accept": "application/json"}


@pytest.mark.live
def test_live_intake_over_testclient(client):
    # 1) start a session
    start = client.post("/session").json()
    sid = start["sessionId"]
    assert start["disclaimer"]

    # 2) a benign opening message — streamed; assert a non-empty reply + a present strip
    resp = client.post(
        f"/session/{sid}/message",
        json={"text": "I've had a mild tension headache for two days"},
        headers=SSE,
    )
    assert resp.status_code == 200
    events = parse_sse(resp.text)
    assert events[-1][0] == "turn"
    turn = events[-1][1]
    assert turn["content"].strip(), "the assistant reply should be non-empty"
    assert turn["level"] in {"CLEAR", "URGENT"}
    assert "ruleId" in turn["strip"]
    print(f"\n[live] benign turn: level={turn['level']} model={turn['model']}")

    # 3) a danger message — the deterministic gate short-circuits to EMERGENCY (no model call)
    resp = client.post(
        f"/session/{sid}/message",
        json={"text": "Now I have crushing chest pain spreading to my left arm and I'm sweating"},
        headers=JSON,
    )
    out = resp.json()
    assert out["level"] == "EMERGENCY"
    assert out["status"] == "halted"
    assert out["emergency"]["actions"][0]["href"] == "tel:911"
    assert out["strip"]["toolsNote"].startswith("gate short-circuited")

    # 4) the trace endpoint reports the (non-zero) cost of the benign model turn
    trace = client.get(f"/session/{sid}/trace").json()
    assert trace["rows"]
    print(
        f"[live] session cost={trace['totalCostUsd']:.4f} "
        f"label={trace['traceCostLabel']!r} rows={len(trace['rows'])}"
    )
    print("[live] turn payload keys:", json.dumps(sorted(turn.keys())))

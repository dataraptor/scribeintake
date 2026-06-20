"""Split 11 wiring: the API serves the static frontend + Proof artifacts, and the turn payload
carries the ``openSlots`` field the frontend binds. Deterministic (no key, orchestrator mocked).
"""

from __future__ import annotations

from .conftest import make_clear_turn


# ------------------------------------------------------------------------------ static frontend
def test_serves_index_at_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    # the host page boots the DC runtime + the data layer
    assert "support.js" in r.text
    assert "api-client.js" in r.text


def test_serves_frontend_assets(client):
    for path in (
        "/index.html",
        "/api-client.js",
        "/support.js",
        "/ScribeIntake.dc.html",
        "/vendor/react.production.min.js",
        "/vendor/react-dom.production.min.js",
    ):
        assert client.get(path).status_code == 200, path


def test_api_routes_win_over_static_mount(client):
    # the static mount is at "/" but explicit API routes still resolve first
    assert client.get("/health").status_code == 200
    assert client.post("/session").status_code == 200


# ------------------------------------------------------------------------------ proof artifacts
def test_proof_leaderboard_served(client):
    r = client.get("/proof/leaderboard.json")
    assert r.status_code == 200
    body = r.json()
    assert "metrics" in body and isinstance(body["metrics"], list)
    groups = {m["group"] for m in body["metrics"]}
    assert {"deterministic", "distributional"} <= groups


def test_proof_cost_report_served(client):
    r = client.get("/proof/cost_report.json")
    assert r.status_code == 200
    assert "trace_cost_label" in r.json()


def test_proof_unknown_artifact_404(client):
    r = client.get("/proof/bogus.json")
    assert r.status_code == 404
    assert r.json()["error"] == "proof_not_found"


# --------------------------------------------------------------- the open_slots wire field (§3.2)
def test_turn_response_carries_open_slots(client, session_id, monkeypatch):
    """The frontend marks a slot filled when it's *not* in openSlots; the turn must expose it."""
    turn = make_clear_turn(session_id)
    turn.open_slots = ["hpi.severity", "medications"]
    monkeypatch.setattr("api.main.run_turn", lambda sid, text, *, conn: turn)
    r = client.post(
        f"/session/{session_id}/message",
        json={"text": "it started this morning"},
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["openSlots"] == ["hpi.severity", "medications"]

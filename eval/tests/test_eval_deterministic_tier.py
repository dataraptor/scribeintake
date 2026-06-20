"""The CI-gated deterministic tier (no API key) — this *is* the per-commit gate.

Proves ``run --deterministic-only`` completes with no key, reports the four gated metrics, and
**fails** the moment a deliberately-broken run violates the triage floor or schema. Acceptance
#5 hangs on this test, so it asserts the gate hard.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.models import ScenarioRun, TurnRecord
from eval.run import main, run_deterministic
from eval.scenario import ScenarioCategory, load_scenarios
from scribeintake.models import EscalationLevel, EscalationSource, TriageBand

SCENARIOS_DIR = str(Path(__file__).resolve().parents[1] / "scenarios")


def _completed_run(sid: str, *, floor: TriageBand, band: TriageBand, soap: dict) -> ScenarioRun:
    turn = TurnRecord(
        turn=1,
        user_msg="x",
        escalation=EscalationLevel.URGENT,
        escalation_source=EscalationSource.gate,
        triage_floor=floor,
        status="completed",
    )
    return ScenarioRun(
        scenario_id=sid,
        category=ScenarioCategory.urgent,
        seed_label="run-1",
        turns=[turn],
        final_soap=soap,
        predicted_band=band,
        final_status="completed",
        n_turns_run=1,
    )


def test_deterministic_tier_passes_with_no_runs_and_no_key():
    scenarios = load_scenarios(SCENARIOS_DIR)
    lb, ok, failures = run_deterministic(scenarios, ts="t")
    assert ok is True, failures
    gated = [m for m in lb.metrics if m.group == "deterministic"]
    assert len(gated) == 4
    assert all(m.passing for m in gated)
    assert {m.key for m in gated} == {
        "rule_correctness",
        "frozen_must_escalate",
        "triage_floor_never_violated",
        "schema_validity",
    }


def test_broken_fixture_violating_triage_floor_fails_the_gate():
    scenarios = load_scenarios(SCENARIOS_DIR)
    from scribeintake.models import SOAP, Subjective

    valid_soap = SOAP(subjective=Subjective(chief_complaint="x")).model_dump()
    broken = _completed_run(
        "floor_breaker",
        floor=TriageBand.gp_urgent,
        band=TriageBand.self_care,  # below the floor — a hard regression
        soap=valid_soap,
    )
    lb, ok, failures = run_deterministic(scenarios, runs=[broken], ts="t")
    assert ok is False
    assert "Triage floor never violated" in failures


def test_broken_fixture_with_invalid_soap_fails_the_gate():
    scenarios = load_scenarios(SCENARIOS_DIR)
    broken = _completed_run(
        "schema_breaker",
        floor=TriageBand.gp_urgent,
        band=TriageBand.gp_urgent,
        soap={"this": "is not a SOAP"},
    )
    lb, ok, failures = run_deterministic(scenarios, runs=[broken], ts="t")
    assert ok is False
    assert "Schema validity" in failures


def test_cli_deterministic_only_writes_leaderboard_and_exits_zero(tmp_path):
    rc = main(
        [
            "--deterministic-only",
            "--scenarios-dir",
            SCENARIOS_DIR,
            "--out-dir",
            str(tmp_path),
            "--ts",
            "20260620T000000Z",
        ]
    )
    assert rc == 0
    md = tmp_path / "leaderboard.md"
    js = tmp_path / "leaderboard.json"
    assert md.exists() and js.exists()
    data = json.loads(js.read_text(encoding="utf-8"))
    # Frontend Proof-tab contract: two groups, camelCase keys, four gated rows.
    assert {r["label"] for r in data["ldDet"]} == {
        "Rule correctness",
        "Frozen must-escalate",
        "Triage floor never violated",
        "Schema validity",
    }
    assert data["meta"]["deterministic_only"] is True
    # Distributional cells are honestly pending (no LLM ran).
    assert all("pending" in r["value"].lower() for r in data["ldDist"])

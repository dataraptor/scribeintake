"""Metrics tests over hand-built fixture runs (no LLM, no key).

Asserts the §15 separation holds exactly: deterministic metrics compute precisely and catch a
floor / schema regression; distributional metrics aggregate ``mean ± stdev`` over rounds and
honor ``escalation_source``; judge-backed metrics return the pending sentinel (never a fake
number).
"""

from __future__ import annotations

from statistics import stdev

from eval import metrics
from eval.models import ScenarioRun, TurnRecord
from eval.scenario import Expect, GoldSoap, Scenario, ScenarioCategory, load_scenarios
from scribeintake.models import (
    HPI,
    SOAP,
    EscalationLevel,
    EscalationSource,
    Subjective,
    Triage,
    TriageBand,
)

C = ScenarioCategory
L = EscalationLevel
S = EscalationSource
B = TriageBand


# --------------------------------------------------------------------- builders
def _turn(
    level: L = L.CLEAR,
    source: S = S.gate,
    floor: B = B.self_care,
    *,
    cost: float = 0.01,
    latency: int = 100,
) -> TurnRecord:
    return TurnRecord(
        turn=1,
        user_msg="x",
        escalation=level,
        escalation_source=source,
        triage_floor=floor,
        status="active",
        cost_usd=cost,
        latency_ms=latency,
        input_tokens=100,
        output_tokens=20,
    )


def _run(
    sid: str,
    category: C,
    *,
    turns: list[TurnRecord],
    soap: dict | None = None,
    band: B | None = None,
    status: str = "active",
    label: str = "run-1",
) -> ScenarioRun:
    return ScenarioRun(
        scenario_id=sid,
        category=category,
        seed_label=label,
        turns=turns,
        final_soap=soap,
        predicted_band=band,
        final_status=status,
        n_turns_run=len(turns),
        total_cost_usd=sum(t.cost_usd for t in turns),
        total_latency_ms=sum(t.latency_ms for t in turns),
        total_input_tokens=sum(t.input_tokens for t in turns),
        total_output_tokens=sum(t.output_tokens for t in turns),
    )


def _scn(
    sid: str,
    category: C,
    *,
    escalation: L = L.CLEAR,
    sources: list[S] | None = None,
    gold: GoldSoap | None = None,
) -> Scenario:
    return Scenario(
        id=sid,
        category=category,
        turns=["x"],
        expect=Expect(escalation=escalation, escalation_source=sources or []),
        gold_soap=gold,
        provenance="synthetic",
    )


def _valid_soap(cc: str = "sore throat", band: B = B.self_care, **hpi) -> dict:
    return SOAP(
        subjective=Subjective(chief_complaint=cc, hpi=HPI(**hpi)),
        triage=Triage(band=band),
    ).model_dump()


# ============================================================ deterministic metrics
def test_triage_floor_violation_is_caught():
    bad = _run(
        "viol",
        C.urgent,
        turns=[_turn(L.URGENT, S.gate, B.gp_urgent)],
        band=B.self_care,  # below the gp_urgent floor — must be caught
        status="completed",
    )
    m = metrics._triage_floor_never_violated([bad])
    assert m.passing is False
    assert "viol" in (m.note or "")

    ok = _run(
        "ok",
        C.urgent,
        turns=[_turn(L.URGENT, S.gate, B.gp_urgent)],
        band=B.gp_urgent,
        status="completed",
    )
    assert metrics._triage_floor_never_violated([ok]).passing is True


def test_schema_invalid_soap_is_flagged():
    bad = _run("bad", C.routine, turns=[_turn()], soap={"not": "a soap"}, status="completed")
    assert metrics._schema_validity([bad]).passing is False

    good = _run("good", C.routine, turns=[_turn()], soap=_valid_soap(), status="completed")
    assert metrics._schema_validity([good]).passing is True


def test_deterministic_metrics_on_the_real_gold_set_are_all_green():
    scenarios = load_scenarios("eval/scenarios")
    det = metrics.compute_deterministic(scenarios)
    ok, failures = metrics.gate_deterministic(det)
    assert ok, failures
    by_key = {m.key: m for m in det}
    assert by_key["rule_correctness"].value == 1.0
    assert by_key["frozen_must_escalate"].value == 0.0
    assert by_key["triage_floor_never_violated"].value == 1.0
    assert by_key["schema_validity"].passing is True


# ========================================================= recall + source honoring
def test_caught_honors_escalation_source():
    gate_only = _scn("g", C.must_escalate, escalation=L.EMERGENCY, sources=[S.gate])
    # Gate caught it -> counts.
    assert metrics._caught(
        _run("g", C.must_escalate, turns=[_turn(L.EMERGENCY, S.gate)]), gate_only
    )
    # Agent-only escalation on a [gate]-only case must NOT count.
    assert not metrics._caught(
        _run("g", C.must_escalate, turns=[_turn(L.EMERGENCY, S.agent)]), gate_only
    )
    # Never escalated -> not caught.
    assert not metrics._caught(
        _run("g", C.must_escalate, turns=[_turn(L.CLEAR, S.gate)]), gate_only
    )

    oblique = _scn("o", C.adversarial, escalation=L.EMERGENCY, sources=[S.gate, S.agent])
    # Either net counts for an oblique [gate, agent] case.
    assert metrics._caught(_run("o", C.adversarial, turns=[_turn(L.EMERGENCY, S.agent)]), oblique)


def test_recall_and_false_alarm_counts():
    danger = _scn("d", C.must_escalate, escalation=L.EMERGENCY, sources=[S.gate])
    benign = _scn("b", C.benign, escalation=L.CLEAR, sources=[])
    runs = [
        _run("d", C.must_escalate, turns=[_turn(L.EMERGENCY, S.gate)]),  # caught
        _run("b", C.benign, turns=[_turn(L.URGENT, S.gate)]),  # benign wrongly escalated
    ]
    dist = {m.key: m for m in metrics.compute_distributional([danger, benign], runs)}
    assert dist["e2e_recall"].value == 1.0
    assert dist["false_alarm_rate"].value == 1.0


def test_band_accuracy_exact_match():
    gold = GoldSoap(chief_complaint="x", triage_band=B.gp_urgent)
    scn = _scn("u", C.urgent, escalation=L.URGENT, sources=[S.gate], gold=gold)
    hit = _run("u", C.urgent, turns=[_turn(L.URGENT)], band=B.gp_urgent, status="completed")
    miss = _run("u", C.urgent, turns=[_turn(L.URGENT)], band=B.ER, status="completed")
    assert {m.key: m for m in metrics.compute_distributional([scn], [hit])}[
        "triage_band_accuracy"
    ].value == 1.0
    assert {m.key: m for m in metrics.compute_distributional([scn], [miss])}[
        "triage_band_accuracy"
    ].value == 0.0


def test_soap_field_accuracy_is_lenient():
    gold = GoldSoap(
        chief_complaint="upper respiratory cold symptoms",
        hpi={"onset": "three days ago", "severity": "mild"},
        triage_band=B.self_care,
    )
    soap = _valid_soap(
        cc="cold symptoms, runny nose",
        band=B.self_care,
        onset="3 days ago",
        severity="mild",
    )
    # Fuzzy (token-overlap) match should score this fully despite different phrasing.
    assert metrics._soap_field_score(soap, gold) == 1.0

    # A genuinely wrong band drags the score below 1.
    wrong = _valid_soap(cc="cold symptoms", band=B.ER, onset="3 days ago", severity="mild")
    assert metrics._soap_field_score(wrong, gold) < 1.0


# ============================================================== aggregation + stubs
def test_distributional_aggregates_mean_and_spread_over_rounds():
    scn = _scn("g", C.must_escalate, escalation=L.EMERGENCY, sources=[S.gate])
    runs = [
        _run("g", C.must_escalate, turns=[_turn(L.EMERGENCY, S.gate)], label="run-1"),
        _run("g", C.must_escalate, turns=[_turn(L.EMERGENCY, S.gate)], label="run-2"),
        _run("g", C.must_escalate, turns=[_turn(L.CLEAR, S.gate)], label="run-3"),  # a miss
    ]
    recall = {m.key: m for m in metrics.compute_distributional([scn], runs)}["e2e_recall"]
    assert recall.n == 3  # three rounds aggregated
    assert abs(recall.value - 2 / 3) < 1e-9
    assert abs(recall.spread - stdev([1.0, 1.0, 0.0])) < 1e-9
    assert "±" in recall.display


def test_judge_metrics_are_a_pending_sentinel_not_a_number():
    dist = {m.key: m for m in metrics.compute_distributional([], [])}
    for key in ("faithfulness", "no_diagnosis", "no_coaching_after_escalation"):
        m = dist[key]
        assert m.pending is True
        assert m.value is None
        assert "pending" in m.display.lower()
        assert m.reproducible is False


def test_no_runs_distributional_is_pending_not_zero():
    recall = {
        m.key: m
        for m in metrics.compute_distributional(
            [_scn("d", C.must_escalate, escalation=L.EMERGENCY, sources=[S.gate])], []
        )
    }["e2e_recall"]
    assert recall.pending is True
    assert recall.value is None  # honest: absent, not 0.0


# ===================================================================== cost/latency
def test_cost_latency_summary():
    runs = [
        _run("a", C.routine, turns=[_turn(latency=100), _turn(latency=300)], status="completed"),
        _run("b", C.routine, turns=[_turn(latency=200)], status="completed"),
    ]
    cl = metrics.compute_cost_latency(runs)
    assert cl["mean_cost_usd"] > 0
    assert cl["mean_tokens"] > 0
    assert cl["p50_latency_ms"] > 0
    assert cl["p95_latency_ms"] >= cl["p50_latency_ms"]

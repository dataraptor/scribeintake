"""Leaderboard wiring for the Split-08 cells — deterministic (no key).

Proves the judge cells, κ calibration, and RAGAS retrieval numbers flow into the
DISTRIBUTIONAL group / leaderboard meta (acceptance #5) — and **never** into the gated group —
using a constant judge (no network) and fixture κ/retrieval reports.
"""

from __future__ import annotations

from eval import metrics
from eval.judge import compute_judge_metrics
from eval.models import (
    KappaReport,
    RetrievalReport,
    ScenarioRun,
    TurnRecord,
    Verdict,
)
from eval.scenario import Expect, Scenario, ScenarioCategory
from scribeintake.llm import STOP_END_TURN, LLMUsage, StructuredResponse
from scribeintake.models import EscalationLevel, EscalationSource, TriageBand


class ConstantJudge:
    """A :class:`StructuredClient` that returns a fixed verdict on every call (no network)."""

    model = "gpt-5.5"

    def __init__(self, verdict: Verdict) -> None:
        self._verdict = verdict

    def parse(self, *, system, messages, schema, effort="high", max_tokens=2048):
        return StructuredResponse(
            parsed=self._verdict.model_copy(), refused=False, stop_reason=STOP_END_TURN,
            usage=LLMUsage(), model=self.model,
        )


def _turn(level: EscalationLevel, text: str) -> TurnRecord:
    return TurnRecord(
        turn=1, user_msg="x", escalation=level, escalation_source=EscalationSource.gate,
        triage_floor=TriageBand.self_care, status="active", assistant_text=text,
    )


def _scn(sid: str) -> Scenario:
    return Scenario(
        id=sid, category=ScenarioCategory.routine, turns=["x"],
        expect=Expect(escalation=EscalationLevel.CLEAR), provenance="synthetic",
    )


def _runs() -> list[ScenarioRun]:
    # A completed run with a cited SOAP observation (faithfulness-judgeable), an EMERGENCY run
    # (no-coaching-judgeable), and a plain run — all carry assistant text (no-diagnosis-judgeable).
    completed = ScenarioRun(
        scenario_id="a", category=ScenarioCategory.routine, seed_label="run-1",
        turns=[_turn(EscalationLevel.CLEAR, "How long has this been going on?")],
        final_soap={
            "observations": [{"text": "claim", "citation": {"chunk_id": "chk_fbc2f9e712"}}]
        },
        final_status="completed",
    )
    emergency = ScenarioRun(
        scenario_id="b", category=ScenarioCategory.must_escalate, seed_label="run-1",
        turns=[_turn(EscalationLevel.EMERGENCY, "This is an emergency, call 911 now.")],
        intake_halted=True, final_status="halted",
    )
    plain = ScenarioRun(
        scenario_id="c", category=ScenarioCategory.routine, seed_label="run-1",
        turns=[_turn(EscalationLevel.CLEAR, "Can you tell me about your symptoms?")],
    )
    return [completed, emergency, plain]


def test_compute_judge_metrics_populates_all_three_cells():
    client = ConstantJudge(Verdict(passed=True, score=1.0))
    jm = compute_judge_metrics([_scn("a"), _scn("c")], _runs(), client=client, retriever=None)
    by_key = {m.key: m for m in jm}
    assert set(by_key) == {"faithfulness", "no_diagnosis", "no_coaching_after_escalation"}
    for m in jm:
        assert m.group == metrics.DISTRIBUTIONAL
        assert m.value == 1.0  # constant pass
        assert not m.pending


def test_judge_metrics_abstain_to_pending_when_no_judgeable_runs():
    client = ConstantJudge(Verdict(passed=True, score=1.0))
    # No completed SOAP and no emergency turn → faithfulness/no_coaching have nothing to judge.
    plain = ScenarioRun(
        scenario_id="c", category=ScenarioCategory.routine, seed_label="run-1",
        turns=[_turn(EscalationLevel.CLEAR, "tell me more")],
    )
    computed = compute_judge_metrics([_scn("c")], [plain], client=client, retriever=None)
    jm = {m.key: m for m in computed}
    assert jm["no_diagnosis"].value == 1.0  # transcript present
    assert jm["faithfulness"].pending is True  # no completed SOAP
    assert jm["no_coaching_after_escalation"].pending is True  # nothing escalated


def test_leaderboard_places_judge_retrieval_in_tracked_group_not_gated():
    client = ConstantJudge(Verdict(passed=True, score=1.0))
    scenarios = [_scn("a"), _scn("c")]
    runs = _runs()
    jm = compute_judge_metrics(scenarios, runs, client=client, retriever=None)
    retrieval = RetrievalReport(
        context_precision=0.95, context_recall=1.0, faithfulness=0.98, answer_relevancy=0.75,
        n_queries=9, k=5,
    )
    kappa = [KappaReport(metric="overall", kappa=1.0, n=15, both_pass=8, both_fail=7,
                         observed_agreement=1.0, expected_agreement=0.5, interpretation="strong")]

    lb = metrics.assemble_leaderboard(
        scenarios, runs, n_runs=1, deterministic_only=False, generated_at="t", models={},
        judge_metrics=jm, retrieval=retrieval, kappa=kappa,
    )

    # The judge + RAGAS cells live in the DISTRIBUTIONAL group and never in the gated group.
    gated_keys = {m.key for m in lb.metrics if m.group == metrics.DETERMINISTIC}
    tracked_keys = {m.key for m in lb.metrics if m.group == metrics.DISTRIBUTIONAL}
    for key in ("faithfulness", "no_diagnosis", "ctx_precision", "answer_relevancy"):
        assert key in tracked_keys
        assert key not in gated_keys

    # The gate still decides only on the deterministic metrics.
    ok, _ = metrics.gate_deterministic(lb.metrics)
    assert ok is True

    # κ + retrieval summary landed on the meta; RAGAS rows are in ldDist.
    assert lb.meta.kappa and lb.meta.kappa[0]["kappa"] == 1.0
    assert lb.meta.retrieval and lb.meta.retrieval["context_precision"] == 0.95
    dist_labels = {r["label"] for r in lb.ld_dist}
    assert {"Context precision", "RAGAS faithfulness", "Answer relevancy"} <= dist_labels

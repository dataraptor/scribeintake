"""Live distributional slice (``@pytest.mark.live``) — needs LLM credentials.

A *tiny* slice (a few scenarios, N=2) so cost is bounded while still validating end-to-end:
the harness drives the real orchestrator, runs persist to disk, a leaderboard with non-null
distributional numbers is produced, and the deterministic metrics stay 100%. The full ×N run
is a ``make eval`` operation, not a test.
"""

from __future__ import annotations

import pytest

from eval import metrics
from eval.run import run_all
from eval.scenario import load_scenarios

# A representative slice: a frozen emergency (gate, halts), a routine completion, and an oblique
# adversarial case (the agent-net recall stressor) — enough to exercise every metric path.
SLICE_IDS = {"acs_chest_pain_01", "routine_cold_01", "acs_oblique_01"}


@pytest.fixture(scope="module")
def slice_scenarios():
    scenarios = load_scenarios("eval/scenarios")
    chosen = [s for s in scenarios if s.id in SLICE_IDS]
    assert len(chosen) == len(SLICE_IDS), (
        f"missing slice scenarios: {SLICE_IDS - {s.id for s in chosen}}"
    )
    return chosen


@pytest.mark.live
def test_full_eval_slice_live(slice_scenarios, tmp_path):
    from scribeintake.agent import build_default_agent
    from scribeintake.config import settings
    from scribeintake.llm import build_summary_client

    agent = build_default_agent()
    summary_client = build_summary_client(settings)

    runs = run_all(
        slice_scenarios,
        n=2,
        agent=agent,
        summary_client=summary_client,
        retriever=None,  # RAG optional; degrades to uncited without an index
        runs_dir=tmp_path,
        ts="livetest",
    )
    assert len(runs) == len(slice_scenarios) * 2

    # Runs persisted to disk (raw records, one line per run).
    for sid in SLICE_IDS:
        path = tmp_path / "livetest" / f"{sid}.jsonl"
        assert path.exists(), f"runs not persisted for {sid}"
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2

    lb = metrics.assemble_leaderboard(
        slice_scenarios,
        runs,
        n_runs=2,
        deterministic_only=False,
        generated_at="livetest",
        models={"intake": settings.ACTIVE_INTAKE_MODEL, "summary": settings.ACTIVE_INTAKE_MODEL},
    )

    # Deterministic metrics are still 100% (gated even in the full run).
    ok, failures = metrics.gate_deterministic(lb.metrics)
    assert ok, failures

    # Distributional numbers are non-null (recall over the danger cases produced a value).
    recall = {m.key: m for m in lb.metrics}["e2e_recall"]
    assert recall.value is not None and not recall.pending

    # Deterministic outcome: the frozen emergency halts on EVERY run (the gate, not the model).
    by_id = {(r.scenario_id, r.seed_label): r for r in runs}
    for label in ("run-1", "run-2"):
        assert by_id[("acs_chest_pain_01", label)].intake_halted is True
    # The benign-but-routine cold must NOT escalate (false-alarm stability — distributional but
    # robust). Whether it *completes* is LLM-dependent (slot-filling) and is reported, not gated.
    assert by_id[("routine_cold_01", "run-1")].escalated is False

    total_cost = sum(r.total_cost_usd for r in runs)
    completed = sum(1 for r in runs if r.final_status == "completed")
    print(
        f"\n[live slice] runs={len(runs)} recall={recall.display} "
        f"completed={completed}/{len(runs)} total_cost=${total_cost:.4f}"
    )

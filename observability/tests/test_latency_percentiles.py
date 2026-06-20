"""Deterministic tests for latency percentiles + the intake/summary split (Split 09, §18)."""

from observability.latency import latency_report, percentiles
from observability.trace import TraceRow


def _row(tool, latency, model="gpt-5.5"):
    return TraceRow(
        session_id="s", turn=1, tool=tool, model=model,
        input_tokens=100, output_tokens=10, cache_read_tokens=0, cache_creation_tokens=0,
        latency_ms=latency, cost_usd=0.001,
    )


def test_percentiles_known_list():
    # 1..10 → p50 nearest-rank = index round(0.5*9)=4 (0-based) → 5; p95 = index 9 → 10.
    p = percentiles(list(range(1, 11)))
    assert p["p50"] == 5.0
    assert p["p95"] == 10.0


def test_percentiles_empty_is_zero():
    assert percentiles([]) == {"p50": 0.0, "p95": 0.0}


def test_percentiles_single_value():
    assert percentiles([42]) == {"p50": 42.0, "p95": 42.0}


def test_intake_and_summary_reported_separately():
    rows = [
        _row("agent_step", 1000),
        _row("agent_step", 2000),
        _row("agent_step", 3000),
        _row("build_summary", 4000),  # first summary → excluded (compile), annotated
        _row("suggest_triage", 5000),
        _row("retrieve_guideline", 9999, model=None),  # local row → ignored entirely
    ]
    rep = latency_report(rows)
    assert rep.intake_n == 3
    assert rep.intake_p50_ms == 2000.0
    # summary percentiles exclude the FIRST summary call (4000), leaving [5000].
    assert rep.first_summary_ms == 4000.0
    assert rep.summary_n == 1
    assert rep.summary_p50_ms == 5000.0
    # the local retrieve_guideline row never enters latency stats
    assert rep.intake_p95_ms == 3000.0


def test_breach_flagged_when_intake_exceeds_target():
    rows = [_row("agent_step", 8000), _row("agent_step", 9000)]  # well over the 6s p95 target
    rep = latency_report(rows)
    metrics = {b["metric"] for b in rep.breaches}
    assert "intake_p95_ms" in metrics


def test_no_breach_when_within_targets():
    rows = [_row("agent_step", 1500), _row("agent_step", 2500)]
    rep = latency_report(rows)
    assert rep.breaches == []


def test_summary_breach_excludes_first_compile_call():
    # First (compile) summary is slow but must NOT be counted; the second is within budget.
    rows = [_row("build_summary", 15000), _row("build_summary", 4000)]
    rep = latency_report(rows)
    assert rep.first_summary_ms == 15000.0
    assert not any(b["metric"] == "summary_p95_ms" for b in rep.breaches)

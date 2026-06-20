"""Latency percentiles over the trace (Split 09, spec section 18).

The intake loop and the terminal summary call have **separate** budgets, so they are reported
separately: intake per-turn p50 < ~3 s / p95 < ~6 s; the summary call ~3–8 s. The first
structured-output call of the day pays a one-time schema-compile cost (spec section 7) — it is
annotated, not flagged as a breach, so it isn't misread as a regression.
"""

from __future__ import annotations

from scribeintake.config import (
    LATENCY_TARGET_INTAKE_P50_MS,
    LATENCY_TARGET_INTAKE_P95_MS,
    LATENCY_TARGET_SUMMARY_MS,
)

from .models import LatencyReport
from .trace import TraceRow

# Tools whose latency belongs to the terminal structured-output call (its own budget).
SUMMARY_TOOLS = frozenset({"build_summary", "suggest_triage"})
# The intake loop's per-model-call tool name (Split 03 trace convention).
INTAKE_TOOL = "agent_step"


def percentiles(latencies: list[int] | list[float], qs: tuple[float, ...] = (0.50, 0.95)) -> dict:
    """Nearest-rank percentiles of a latency sample. Empty → all 0.0.

    Returns ``{"p50": …, "p95": …}`` (keys derived from ``qs``). Pure / deterministic.
    """
    vals = sorted(float(v) for v in latencies)
    out: dict[str, float] = {}
    for q in qs:
        out[f"p{int(round(q * 100))}"] = _percentile(vals, q)
    return out


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return float(sorted_vals[idx])


def latency_report(rows: list[TraceRow]) -> LatencyReport:
    """Split trace latencies into intake vs summary and report p50/p95 + breaches (section 18).

    A model-call trace row with a positive ``latency_ms`` counts; local ``$0`` tool rows are
    ignored (their latency is local I/O, not a model budget). The **first** summary call is
    excluded from the summary percentiles and surfaced as ``first_summary_ms`` (schema-compile).
    """
    intake = [r.latency_ms for r in rows if r.tool == INTAKE_TOOL and r.latency_ms]
    summary_rows = [r for r in rows if r.tool in SUMMARY_TOOLS and r.latency_ms]

    first_summary_ms: float | None = None
    summary_lat = [r.latency_ms for r in summary_rows]
    if summary_rows:
        # Annotate (and exclude from percentiles) the first summary call — schema-compile (§7).
        first_summary_ms = float(summary_rows[0].latency_ms)
        summary_lat = [r.latency_ms for r in summary_rows[1:]]

    ip = percentiles(intake)
    sp = percentiles(summary_lat)

    targets = {
        "intake_p50_ms": LATENCY_TARGET_INTAKE_P50_MS,
        "intake_p95_ms": LATENCY_TARGET_INTAKE_P95_MS,
        "summary_ms": LATENCY_TARGET_SUMMARY_MS,
    }
    breaches: list[dict] = []

    def _flag(metric: str, value: float, target: int) -> None:
        breaches.append({"metric": metric, "value": value, "target": target})

    if intake and ip["p50"] > LATENCY_TARGET_INTAKE_P50_MS:
        _flag("intake_p50_ms", ip["p50"], targets["intake_p50_ms"])
    if intake and ip["p95"] > LATENCY_TARGET_INTAKE_P95_MS:
        _flag("intake_p95_ms", ip["p95"], targets["intake_p95_ms"])
    # Summary percentiles exclude the first (compile) call, so a breach here is a real regression.
    if summary_lat and sp["p95"] > LATENCY_TARGET_SUMMARY_MS:
        _flag("summary_p95_ms", sp["p95"], targets["summary_ms"])

    return LatencyReport(
        intake_n=len(intake),
        intake_p50_ms=ip["p50"],
        intake_p95_ms=ip["p95"],
        summary_n=len(summary_lat),
        summary_p50_ms=sp["p50"],
        summary_p95_ms=sp["p95"],
        first_summary_ms=first_summary_ms,
        targets=targets,
        breaches=breaches,
    )

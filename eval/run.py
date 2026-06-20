"""Eval runner CLI (spec section 15) — ``python -m eval.run``.

Two tiers, deliberately separate code paths:

* ``--deterministic-only`` — the **per-commit CI gate**. Loads the gold set, computes the four
  gated (no-LLM) metrics, writes the leaderboard with the distributional cells marked *pending*,
  and **exits non-zero on any regression** (rule correctness, frozen must-escalate, triage
  floor, schema validity). No API key, no model load.
* default (``--n N``) — the **full distributional run**. Drives every scenario end-to-end ×N
  through the real orchestrator, persists **every** run to ``eval/runs/<ts>/*.jsonl``, and
  reports the distributional metrics as mean ± spread. Needs LLM credentials.

The runner imports the harness (in-process); it never talks to an HTTP service.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from scribeintake.config import MODEL_SUMMARY, settings

from . import metrics
from .harness import run_scenario
from .models import Leaderboard, ScenarioRun
from .scenario import Scenario, load_scenarios

_PKG = Path(__file__).resolve().parent
DEFAULT_SCENARIOS = _PKG / "scenarios"
DEFAULT_RUNS_DIR = _PKG / "runs"
DEFAULT_OUT_DIR = _PKG


def _models() -> dict[str, str]:
    return {
        "intake": settings.ACTIVE_INTAKE_MODEL,
        "summary": settings.ACTIVE_INTAKE_MODEL,
        "spec_summary": MODEL_SUMMARY,
    }


def _now_ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# ----------------------------------------------------------------- full (live) run
def run_all(
    scenarios: list[Scenario],
    *,
    n: int,
    agent: object | None = None,
    summary_client: object | None = None,
    retriever: object | None = None,
    runs_dir: Path | None = None,
    ts: str | None = None,
    progress: bool = False,
) -> list[ScenarioRun]:
    """Run every scenario ×N (sequential — hermetic per run) and persist each run as JSONL.

    Sequential is fine for v1: each :func:`run_scenario` is hermetic (its own SQLite), so this
    is correctness-safe to parallelize later if needed. Persisting happens per scenario so a
    long run is never lost wholesale.
    """
    ts = ts or _now_ts()
    out_dir = (runs_dir or DEFAULT_RUNS_DIR) / ts
    runs: list[ScenarioRun] = []
    for i, scenario in enumerate(scenarios, start=1):
        scenario_runs = [
            run_scenario(
                scenario,
                seed_label=f"run-{k}",
                agent=agent,
                summary_client=summary_client,
                retriever=retriever,
            )
            for k in range(1, n + 1)
        ]
        _persist(scenario_runs, out_dir)
        runs.extend(scenario_runs)
        if progress:
            esc = scenario_runs[0].max_level.value
            print(f"  [{i}/{len(scenarios)}] {scenario.id}: {esc} (×{n})", file=sys.stderr)
    return runs


def _persist(scenario_runs: list[ScenarioRun], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not scenario_runs:
        return
    path = out_dir / f"{scenario_runs[0].scenario_id}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for run in scenario_runs:
            fh.write(run.model_dump_json() + "\n")


# --------------------------------------------------------- deterministic (CI gate)
def run_deterministic(
    scenarios: list[Scenario],
    *,
    runs: list[ScenarioRun] | None = None,
    ts: str | None = None,
) -> tuple[Leaderboard, bool, list[str]]:
    """Compute + gate the deterministic tier. Returns ``(leaderboard, ok, failures)``.

    ``runs`` is normally ``None`` (the CI path runs no models); tests pass a deliberately-broken
    run to prove the gate fails on a triage-floor / schema violation.
    """
    det = metrics.compute_deterministic(scenarios, runs)
    ok, failures = metrics.gate_deterministic(det)
    lb = metrics.assemble_leaderboard(
        scenarios,
        runs or [],
        n_runs=0,
        deterministic_only=True,
        generated_at=ts or _now_ts(),
        models=_models(),
    )
    return lb, ok, failures


# -------------------------------------------------------------------- leaderboard
def write_leaderboard(lb: Leaderboard, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "leaderboard.json"
    md_path = out_dir / "leaderboard.md"
    json_path.write_text(lb.model_dump_json(by_alias=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(lb), encoding="utf-8")
    return json_path, md_path


def render_markdown(lb: Leaderboard) -> str:
    m = lb.meta
    lines = [
        "# ScribeIntake — Eval Leaderboard",
        "",
        f"> {lb.framing}",
        "",
        f"- **Generated:** {m.generated_at}",
        f"- **Scenarios:** {m.scenario_count} · **N per scenario:** {m.n_runs} · "
        f"**rounds aggregated:** {m.rounds}",
        f"- **Models:** intake `{m.models.get('intake', '—')}` · "
        f"summary `{m.models.get('summary', '—')}`",
        f"- **Tier:** {'deterministic-only (no API key)' if m.deterministic_only else 'full'}",
        "",
        "## DETERMINISTIC · GATED (100%, per-commit, no API key)",
        "",
        "| Metric | Value | Status |",
        "|---|---|---|",
    ]
    for row in (mv for mv in lb.metrics if mv.group == metrics.DETERMINISTIC):
        status = "✅" if row.passing else "❌ REGRESSION"
        lines.append(f"| {row.label} | {row.display} | {status} |")

    lines += [
        "",
        "## DISTRIBUTIONAL · TRACKED, NOT GATED (mean ± spread over N runs)",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for row in (mv for mv in lb.metrics if mv.group == metrics.DISTRIBUTIONAL):
        lines.append(f"| {row.label} | {row.display} |")

    if not m.deterministic_only and m.mean_cost_usd is not None:
        lines += [
            "",
            "## Cost & latency (tracked, informational)",
            "",
            f"- **Mean $/session:** ${m.mean_cost_usd:.4f}",
            f"- **Mean tokens/session:** {m.mean_tokens:.0f}",
            f"- **Per-turn latency p50/p95:** {m.p50_latency_ms:.0f} ms"
            f" / {m.p95_latency_ms:.0f} ms",
        ]
    lines.append("")
    return "\n".join(lines)


def _print_summary(lb: Leaderboard, ok: bool, failures: list[str]) -> None:
    print(lb.framing)
    print()
    print("DETERMINISTIC · GATED:")
    for row in (mv for mv in lb.metrics if mv.group == metrics.DETERMINISTIC):
        mark = "OK" if row.passing else "FAIL"
        note = f"  ({row.note})" if row.note else ""
        print(f"  [{mark}] {row.label}: {row.display}{note}")
    print()
    print("DISTRIBUTIONAL · TRACKED:")
    for row in (mv for mv in lb.metrics if mv.group == metrics.DISTRIBUTIONAL):
        print(f"  - {row.label}: {row.display}")
    if not ok:
        print()
        print(f"DETERMINISTIC GATE FAILED: {failures}", file=sys.stderr)


# --------------------------------------------------------------------------- CLI
def _filter(
    scenarios: list[Scenario], *, category: str | None, max_n: int | None
) -> list[Scenario]:
    if category:
        scenarios = [s for s in scenarios if s.category.value == category]
    if max_n is not None:
        scenarios = scenarios[:max_n]
    return scenarios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ScribeIntake eval runner")
    parser.add_argument("--n", type=int, default=3, help="runs per scenario (default 3)")
    parser.add_argument("--category", default=None, help="restrict to one scenario category")
    parser.add_argument("--max", type=int, default=None, help="cap the number of scenarios")
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="the CI gate: gated metrics only, no API key, no model load",
    )
    parser.add_argument("--scenarios-dir", default=str(DEFAULT_SCENARIOS))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--ts", default=None, help="override the run timestamp (reproducible)")
    args = parser.parse_args(argv)

    scenarios = _filter(load_scenarios(args.scenarios_dir), category=args.category, max_n=args.max)
    out_dir = Path(args.out_dir)

    if args.deterministic_only:
        lb, ok, failures = run_deterministic(scenarios, ts=args.ts)
        write_leaderboard(lb, out_dir)
        _print_summary(lb, ok, failures)
        return 0 if ok else 1

    # Full distributional run — build the live clients once, reuse across runs.
    from scribeintake.agent import build_default_agent
    from scribeintake.llm import build_summary_client

    agent = build_default_agent()
    summary_client = build_summary_client(settings)
    retriever = _maybe_retriever()

    ts = args.ts or _now_ts()
    runs = run_all(
        scenarios,
        n=args.n,
        agent=agent,
        summary_client=summary_client,
        retriever=retriever,
        runs_dir=Path(args.runs_dir),
        ts=ts,
        progress=True,
    )
    lb = metrics.assemble_leaderboard(
        scenarios,
        runs,
        n_runs=args.n,
        deterministic_only=False,
        generated_at=ts,
        models=_models(),
    )
    _, ok, failures = (lb, *metrics.gate_deterministic(lb.metrics))
    write_leaderboard(lb, out_dir)
    _print_summary(lb, ok, failures)
    # Even the full run hard-fails on a deterministic regression (those are still gated).
    return 0 if ok else 1


def _maybe_retriever() -> object | None:
    """Best-effort live retriever; ``None`` if no index is built (degrades to uncited)."""
    try:
        from scribeintake.rag import get_retriever

        return get_retriever()
    except Exception:  # noqa: BLE001 - unbuilt/unreadable index is non-fatal for the eval
        return None


if __name__ == "__main__":
    raise SystemExit(main())

"""The published cost/observability report (Split 09, spec section 16).

Assembles a :class:`CostReport` from the trace (the ``tool_calls`` table for the rich cache-aware
numbers, optionally the eval ``runs/`` fleet for the cost trend) and serialises it to
``cost_report.{json,md}`` — the artifact the README (Split 12) and the frontend Proof tab
(Split 11) consume. Reads existing data only; no model call, no service.

When no real trace exists yet, :func:`seed_synthetic_db` populates a small, clearly-labelled
synthetic trace so the artifact + dashboard always render key-free (``source = "synthetic-demo"``).
The headline live numbers are produced by running a real intake first, then this report over
``data/scribeintake.db`` (``source = "live-db"``).
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

from scribeintake import db

from .cost import breakdown, fleet_cost
from .latency import latency_report
from .models import CostReport
from .trace import TraceRow, read_tool_calls

_PKG = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = _PKG
DEFAULT_RUNS_DIR = _PKG.parent / "eval" / "runs"


def _trace_cost_label(total: float, pct_saved: float) -> str:
    """The mockup's ``traceCost`` string: ``"$0.012 · cache 61% saved"``."""
    return f"${total:.3f} · cache {round(pct_saved * 100)}% saved"


def build_report(
    rows: list[TraceRow],
    *,
    generated_at: str,
    source: str,
    models: dict[str, str] | None = None,
    runs_dir: str | Path | None = None,
) -> CostReport:
    """Assemble a :class:`CostReport` from already-read trace rows."""
    cost = breakdown(rows, scope=source)
    lat = latency_report(rows)
    tool_usage = dict(Counter(r.tool for r in rows))

    fleet = None
    notes: list[str] = []
    if runs_dir is not None:
        fc = fleet_cost(runs_dir)
        if fc.n_sessions:
            fleet = fc
            notes.append(
                f"Fleet trend over {fc.n_sessions} eval session(s); cache savings are measured on "
                "the live DB (per-run records do not carry cache buckets)."
            )

    if cost.local_tool_cost_usd:
        notes.append(
            f"WARNING: local tool rows reported ${cost.local_tool_cost_usd:.4f} (expected $0)."
        )
    notes.append(
        "No-cache baseline = observed cached tokens repriced at full input price "
        "(pricing.no_cache_cost_usd); the wired GPT-5.5 prefix cache is automatic, so this exact "
        "counterfactual is more rigorous than re-running."
    )

    return CostReport(
        generated_at=generated_at,
        source=source,
        models=models or {},
        cost=cost,
        savings=cost.savings,
        latency=lat,
        fleet=fleet,
        tool_usage=tool_usage,
        cache_hit_rate=cost.savings.cache_read_share,
        trace_cost_label=_trace_cost_label(cost.total_cost_usd, cost.savings.pct_saved),
        notes=notes,
    )


def report_from_db(
    conn: sqlite3.Connection,
    *,
    generated_at: str,
    source: str = "live-db",
    models: dict[str, str] | None = None,
    runs_dir: str | Path | None = None,
) -> CostReport:
    """Build the report over **all** sessions in a DB."""
    return build_report(
        read_tool_calls(conn),
        generated_at=generated_at,
        source=source,
        models=models,
        runs_dir=runs_dir,
    )


# ----------------------------------------------------------------- synthetic seed
def seed_synthetic_db(conn: sqlite3.Connection, *, session_id: str = "demo-session") -> str:
    """Insert a small, realistic, **clearly-synthetic** trace so the artifact renders key-free.

    Models a 3-turn intake: turn 1 is cache-cold; turns 2–3 read a stable prefix; finalisation
    runs the two terminal structured-output calls + a free local RAG row. Token/latency figures
    are representative of a real GPT-5.5 intake (cf. the Split 07 live run: ~$0.012/session).
    """
    from scribeintake.models import ToolCallTrace

    db.init_db(conn)
    conn.execute(
        "INSERT INTO sessions (id, started_at, status) VALUES (?, ?, ?)",
        (session_id, "2026-06-20T00:00:00Z", "completed"),
    )
    model = "gpt-5.5"
    # (turn, tool, model, input, output, cache_read, latency_ms)
    plan = [
        (1, "agent_step", model, 1850, 40, 0, 2100),  # cold: prefix not yet cached
        (1, "record_intake", None, 0, 0, 0, 3),  # local tool, $0
        (1, "retrieve_guideline", None, 0, 0, 0, 5),  # local RAG, $0
        (2, "agent_step", model, 260, 36, 1792, 1700),  # warm: most of the prefix from cache
        (2, "record_intake", None, 0, 0, 0, 3),
        (3, "agent_step", model, 300, 34, 1856, 1650),  # warm
        (3, "record_intake", None, 0, 0, 0, 3),
        (3, "build_summary", model, 520, 430, 1792, 4200),  # terminal SOAP (first = compile-ish)
        (3, "suggest_triage", model, 480, 120, 980, 2300),  # terminal triage
    ]
    from scribeintake import pricing

    for turn, tool, mdl, it, ot, cr, lat in plan:
        cost = pricing.cost_usd(mdl, it, ot, 0, cr) if mdl in pricing.PRICES else 0.0
        tr = ToolCallTrace(
            session_id=session_id,
            turn=turn,
            tool=tool,
            model=mdl,
            input_tokens=it,
            output_tokens=ot,
            cache_read_tokens=cr,
            cache_creation_tokens=0,
            latency_ms=lat,
            cost_usd=cost,
            ts="2026-06-20T00:00:00Z",
        )
        db.log_tool_call(conn, tr)
    return session_id


# -------------------------------------------------------------------- write / render
def write_report(report: CostReport, out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``cost_report.json`` + ``cost_report.md``; returns both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "cost_report.json"
    md_path = out / "cost_report.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def render_markdown(r: CostReport) -> str:
    """Render the cost report as UTF-8 Markdown (the README / Proof-tab artifact)."""
    s = r.savings
    lat = r.latency
    lines = [
        "# ScribeIntake: Cost & Observability Report",
        "",
        f"> {r.trace_cost_label}",
        "",
        f"- **Generated:** {r.generated_at}",
        f"- **Source:** `{r.source}`"
        + (f" · model `{r.models.get('intake', 'n/a')}`" if r.models else ""),
        f"- **Model calls:** {r.cost.n_model_calls} · **total trace rows:** {r.cost.n_calls}",
        "",
        "## Cost (cache-aware, spec §16)",
        "",
        "| | USD |",
        "|---|---|",
        f"| With prompt caching | ${s.with_cache_usd:.4f} |",
        f"| Without caching (counterfactual) | ${s.no_cache_usd:.4f} |",
        f"| **Saved by caching** | **{s.pct_saved * 100:.0f}%** |",
        f"| Total trace cost | ${r.cost.total_cost_usd:.4f} |",
        f"| Local tool cost (must be $0) | ${r.cost.local_tool_cost_usd:.4f} |",
        "",
        f"Cache-read share of prompt tokens: **{r.cache_hit_rate * 100:.0f}%** "
        f"({s.cache_read_tokens} tokens served from cache).",
        "",
        "### Per-model token totals",
        "",
        "| Model | Calls | Input | Output | Cache-read | Cost |",
        "|---|---|---|---|---|---|",
    ]
    for m in r.cost.per_model:
        lines.append(
            f"| `{m.model}` | {m.calls} | {m.input_tokens} | {m.output_tokens} | "
            f"{m.cache_read_tokens} | ${m.cost_usd:.4f} |"
        )

    lines += [
        "",
        "## Latency percentiles (spec §18)",
        "",
        "| Scope | n | p50 (ms) | p95 (ms) | target |",
        "|---|---|---|---|---|",
        f"| Intake per-turn | {lat.intake_n} | {lat.intake_p50_ms:.0f} | {lat.intake_p95_ms:.0f} "
        f"| p50<{lat.targets.get('intake_p50_ms', 0)} "
        f"/ p95<{lat.targets.get('intake_p95_ms', 0)} |",
        f"| Summary call | {lat.summary_n} | {lat.summary_p50_ms:.0f} | {lat.summary_p95_ms:.0f} "
        f"| <{lat.targets.get('summary_ms', 0)} |",
    ]
    if lat.first_summary_ms is not None:
        lines.append(
            f"\n_First summary call ({lat.first_summary_ms:.0f} ms) excluded from the percentiles "
            "(one-time schema compile, not a regression)._"
        )
    if lat.breaches:
        lines.append("")
        lines.append("**⚠ Latency breaches:** " + ", ".join(
            f"{b['metric']}={b['value']:.0f}ms > {b['target']}ms" for b in lat.breaches
        ))
    else:
        lines.append("\nAll measured latencies are within the §18 targets. ✅")

    lines += ["", "## Tool usage", "", "| Tool | Calls |", "|---|---|"]
    for tool, n in sorted(r.tool_usage.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{tool}` | {n} |")

    if r.fleet:
        f = r.fleet
        lines += [
            "",
            "## Fleet (eval runs/) cost trend",
            "",
            f"- **Sessions:** {f.n_sessions} · **mean $/session:** ${f.mean_cost_usd:.4f} · "
            f"**total:** ${f.total_cost_usd:.4f} · **mean tokens/session:** {f.mean_tokens:.0f}",
        ]

    if r.notes:
        lines += ["", "## Notes", ""]
        lines += [f"- {n}" for n in r.notes]
    lines.append("")
    return "\n".join(lines)

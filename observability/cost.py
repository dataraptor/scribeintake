"""Cache-aware cost accounting over the trace (Split 09, spec section 16).

Every number here is derived from the **token buckets** persisted on each ``tool_calls`` row
(``pricing.cost_usd`` over all four buckets), so prompt caching shows up as a real saving instead
of being erased. The no-cache baseline is a counterfactual repricing of the same tokens at full
input price (``pricing.savings_vs_no_cache``) — exact and noise-free.

Two scopes: :func:`session_cost` (one live session in a SQLite DB, the rich source with cache
buckets) and :func:`fleet_cost` (the eval ``runs/`` fleet — cost/latency trend, no cache buckets,
since per-run records don't carry them). :func:`export_session_jsonl` writes the per-session JSONL
trace export (acceptance #1).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from statistics import mean

from scribeintake import pricing

from .models import CacheSavings, CostBreakdown, FleetCost, ModelTokens
from .trace import TraceRow, iter_run_records, read_tool_calls


def _savings(rows: list[TraceRow]) -> CacheSavings:
    s = pricing.savings_vs_no_cache(r.as_dict() for r in rows)
    return CacheSavings(
        with_cache_usd=s["with_cache"],
        no_cache_usd=s["no_cache"],
        pct_saved=s["pct_saved"],
        cache_read_tokens=int(s["cache_read_tokens"]),
        cache_read_share=s["cache_read_share"],
    )


def breakdown(rows: list[TraceRow], *, scope: str) -> CostBreakdown:
    """Aggregate a list of trace rows into a :class:`CostBreakdown`.

    Model cost is **recomputed** from the buckets via :func:`pricing.cost_usd` (not read from the
    persisted ``cost_usd``) so the report can never silently inherit an input-only mis-pricing —
    the honesty guarantee of spec section 16. Local tool rows (``model`` is ``None``) cost ``$0``.
    """
    per_model: dict[str, ModelTokens] = {}
    per_tool: dict[str, float] = {}
    total = 0.0
    local_total = 0.0
    n_model_calls = 0

    for r in rows:
        if r.is_model_call and r.model in pricing.PRICES:
            cost = pricing.cost_usd(
                r.model,
                r.input_tokens,
                r.output_tokens,
                r.cache_creation_tokens,
                r.cache_read_tokens,
            )
            n_model_calls += 1
            mt = per_model.setdefault(r.model, ModelTokens(model=r.model))
            mt.calls += 1
            mt.input_tokens += r.input_tokens
            mt.output_tokens += r.output_tokens
            mt.cache_read_tokens += r.cache_read_tokens
            mt.cache_creation_tokens += r.cache_creation_tokens
            mt.cost_usd += cost
        else:
            cost = 0.0  # local RAG / tool rows are free
            local_total += r.cost_usd  # should already be 0.0; surfaced for the assertion
        total += cost
        per_tool[r.tool] = per_tool.get(r.tool, 0.0) + cost

    return CostBreakdown(
        scope=scope,
        n_calls=len(rows),
        n_model_calls=n_model_calls,
        total_cost_usd=total,
        local_tool_cost_usd=local_total,
        per_model=list(per_model.values()),
        per_tool_cost_usd=per_tool,
        savings=_savings(rows),
    )


def session_cost(conn: sqlite3.Connection, session_id: str) -> CostBreakdown:
    """Cost breakdown for one session from its ``tool_calls`` rows."""
    return breakdown(read_tool_calls(conn, session_id), scope=f"session:{session_id}")


def db_cost(conn: sqlite3.Connection) -> CostBreakdown:
    """Cost breakdown over **all** sessions in a DB (the whole ``tool_calls`` table)."""
    return breakdown(read_tool_calls(conn), scope="db")


def fleet_cost(runs_dir: str | Path) -> FleetCost:
    """Aggregate $/session + tokens/session over the eval ``runs/`` JSONL fleet.

    Reads the persisted per-run totals (``total_cost_usd`` / ``total_input_tokens`` /
    ``total_output_tokens``). Cache buckets are not in run records, so the cache-savings figure
    comes from :func:`session_cost` over a live DB, not from here.
    """
    costs: list[float] = []
    tokens: list[int] = []
    for rec in iter_run_records(runs_dir):
        costs.append(float(rec.get("total_cost_usd", 0.0)))
        tokens.append(
            int(rec.get("total_input_tokens", 0)) + int(rec.get("total_output_tokens", 0))
        )
    if not costs:
        return FleetCost()
    return FleetCost(
        n_sessions=len(costs),
        total_cost_usd=sum(costs),
        mean_cost_usd=mean(costs),
        mean_tokens=mean(tokens) if tokens else 0.0,
        per_session_cost_usd=costs,
    )


def export_session_jsonl(conn: sqlite3.Connection, session_id: str, out_path: str | Path) -> Path:
    """Write one JSON line per ``tool_calls`` row for a session (the per-session trace export)."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_tool_calls(conn, session_id)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.as_dict()) + "\n")
    return path

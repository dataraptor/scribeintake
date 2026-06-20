"""Observability, cost & prompt-caching analysis (Split 09, spec section 16/18).

A **read-only** analysis layer over the ``tool_calls`` audit table and the eval ``runs/`` JSONL:
cache-aware cost accounting, a no-cache savings baseline, latency percentiles, a cold-vs-warm
prompt-cache verification (live), a static dashboard, and the ``cost_report.{json,md}`` artifact
the README + frontend Proof tab consume. It calls **no model** (except the opt-in live cache
check) and stands up **no service** — the portfolio point is reliability + cost control, not yet
another daemon.

The wired provider is Azure OpenAI GPT-5.5 (Split 03): prefix caching is automatic (no
``cache_control``), there is no cache-**write** bucket, and ``cache_read_tokens`` is the observable
caching signal. The no-cache baseline therefore reprices observed cached tokens at full input
price — see :func:`scribeintake.pricing.no_cache_cost_usd`.
"""

from __future__ import annotations

from .cost import breakdown, db_cost, export_session_jsonl, fleet_cost, session_cost
from .latency import latency_report, percentiles
from .models import (
    CacheCheckResult,
    CacheSavings,
    CostBreakdown,
    CostReport,
    FleetCost,
    LatencyReport,
    ModelTokens,
)

__all__ = [
    "breakdown",
    "db_cost",
    "export_session_jsonl",
    "fleet_cost",
    "session_cost",
    "latency_report",
    "percentiles",
    "CacheCheckResult",
    "CacheSavings",
    "CostBreakdown",
    "CostReport",
    "FleetCost",
    "LatencyReport",
    "ModelTokens",
]

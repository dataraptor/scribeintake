"""Typed shapes for the observability layer (Split 09).

These Pydantic v2 models are what ``observability.cost`` / ``observability.latency`` produce and
what ``observability.report`` serialises to ``cost_report.{json,md}`` — the artifact consumed by
the README (Split 12) and the frontend Proof tab (Split 11). Keep the field names stable; the
Proof tab binds ``trace_cost_label`` (the mockup's ``traceCost = "$… · cache N% saved"``).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelTokens(BaseModel):
    """Per-model token totals across the calls in a scope (one session or a fleet)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0


class CacheSavings(BaseModel):
    """With-cache vs no-cache cost and the measured % saved (spec section 16).

    ``no_cache_usd`` is the counterfactual repricing of the same tokens at full input price
    (``pricing.no_cache_cost_usd``), so ``pct_saved`` is an exact, noise-free measurement.
    """

    model_config = ConfigDict(extra="forbid")

    with_cache_usd: float = 0.0
    no_cache_usd: float = 0.0
    pct_saved: float = 0.0  # fraction in [0,1]
    cache_read_tokens: int = 0
    cache_read_share: float = 0.0  # fraction of prompt tokens served from cache


class CostBreakdown(BaseModel):
    """Cost accounting for one scope (a session or the fleet)."""

    model_config = ConfigDict(extra="forbid")

    scope: str  # "session:<id>" | "fleet"
    n_calls: int = 0
    n_model_calls: int = 0
    total_cost_usd: float = 0.0
    local_tool_cost_usd: float = 0.0  # RAG / local tool rows — must be 0.0 (asserted)
    per_model: list[ModelTokens] = Field(default_factory=list)
    per_tool_cost_usd: dict[str, float] = Field(default_factory=dict)
    savings: CacheSavings = Field(default_factory=CacheSavings)


class FleetCost(BaseModel):
    """Aggregate cost/latency across many sessions (the eval ``runs/`` fleet, section 18)."""

    model_config = ConfigDict(extra="forbid")

    n_sessions: int = 0
    total_cost_usd: float = 0.0
    mean_cost_usd: float = 0.0
    mean_tokens: float = 0.0
    per_session_cost_usd: list[float] = Field(default_factory=list)


class LatencyReport(BaseModel):
    """p50/p95 latency split into the intake loop and the terminal summary call (section 18).

    The intake per-turn budget (p50 < ~3 s, p95 < ~6 s) and the summary budget (~3–8 s) are
    separate. ``first_summary_ms`` is annotated (not counted as a breach): the first
    structured-output call of the day pays a one-time schema-compile cost (spec section 7).
    """

    model_config = ConfigDict(extra="forbid")

    intake_n: int = 0
    intake_p50_ms: float = 0.0
    intake_p95_ms: float = 0.0
    summary_n: int = 0
    summary_p50_ms: float = 0.0
    summary_p95_ms: float = 0.0
    first_summary_ms: float | None = None
    targets: dict[str, int] = Field(default_factory=dict)
    breaches: list[dict] = Field(default_factory=list)


class CacheCheckResult(BaseModel):
    """Outcome of the live cold-vs-warm prompt-cache verification (section 7).

    The demonstration is a back-to-back repeat of the terminal structured-output call
    (``build_summary``): the large, byte-stable ``system + SOAP json_schema`` prefix is cache-cold
    on the first call and **read** on warm repeats. (On this Azure GPT-5.5 deployment the agent
    loop's smaller system+tools prefix does not surface cache hits, but the terminal call's
    schema-dominated prefix reliably does — see the Split 09 session log.)
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    n_calls: int
    cold_cache_read: int  # cache_read tokens on the first (cache-priming) call
    warm_cache_read_max: int  # peak cache_read on a later, warm call
    cache_read_demonstrated: bool  # warm cache_read > 0
    cached_prefix_tokens: int  # the warm call's cached prefix size (tokens served from cache)
    openai_min_cacheable: int  # OpenAI's documented minimum cacheable prefix (~1024)
    clears_openai_min: bool
    sonnet_floor: int  # the spec's notional Claude-Sonnet floor (2048) — reported for the §7 map
    clears_sonnet_floor: bool
    per_call: list[dict] = Field(default_factory=list)
    note: str = ""


class CostReport(BaseModel):
    """The published observability artifact (``cost_report.{json,md}``)."""

    model_config = ConfigDict(extra="forbid")

    generated_at: str
    source: str  # "live-db" | "synthetic-demo" | "fleet" — where the numbers came from
    models: dict[str, str] = Field(default_factory=dict)
    cost: CostBreakdown
    savings: CacheSavings
    latency: LatencyReport
    fleet: FleetCost | None = None
    tool_usage: dict[str, int] = Field(default_factory=dict)  # tool -> call count
    cache_hit_rate: float = 0.0  # fraction of prompt tokens served from cache
    # Pre-formatted for the frontend Proof tab (mockup ``traceCost``).
    trace_cost_label: str = ""
    notes: list[str] = Field(default_factory=list)

"""Cache-aware cost accounting (spec section 16).

Pure and deterministic. Cost is computed from the **three input buckets** plus
output, so prompt caching shows up as a real saving instead of being erased:

* ``input_tokens``                 -> 1.00x input price
* ``cache_creation_input_tokens``  -> 1.25x input price (5-min ephemeral write)
* ``cache_read_input_tokens``      -> 0.10x input price
* ``output_tokens``                -> output price

Prices (USD per MTok, input/output) verified against the claude-api skill 2026-06-20.
"""

from __future__ import annotations

from typing import Any

# (input $/MTok, output $/MTok)
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # Azure OpenAI GPT-5.5 — the provider wired in Split 03. These rates are an
    # ESTIMATE (not list-price-pinned like the Claude rows above); they exist so the
    # cost trace is non-zero and relatively meaningful, not for billing accuracy.
    # OpenAI ``usage`` maps prompt_tokens->input, completion_tokens->output, and
    # prompt_tokens_details.cached_tokens->cache_read (no separate cache-write bucket).
    "gpt-5.5": (1.25, 10.0),
}

CACHE_WRITE_MULTIPLIER = 1.25  # 5-min ephemeral cache write
CACHE_READ_MULTIPLIER = 0.10  # cache read

_PER_MTOK = 1_000_000.0


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Return the USD cost of one call using all four token buckets."""
    if model not in PRICES:
        raise ValueError(f"unknown model for pricing: {model!r}")
    input_price, output_price = PRICES[model]
    in_rate = input_price / _PER_MTOK
    out_rate = output_price / _PER_MTOK
    return (
        input_tokens * in_rate
        + cache_creation_tokens * in_rate * CACHE_WRITE_MULTIPLIER
        + cache_read_tokens * in_rate * CACHE_READ_MULTIPLIER
        + output_tokens * out_rate
    )


def cost_from_usage(model: str, usage: Any) -> float:
    """Compute cost directly from an Anthropic ``usage`` object (or a dict).

    Reads ``input_tokens``, ``output_tokens``, ``cache_creation_input_tokens`` and
    ``cache_read_input_tokens``; missing fields are treated as 0.
    """

    def _get(name: str) -> int:
        raw = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, 0)
        return int(raw or 0)

    return cost_usd(
        model,
        input_tokens=_get("input_tokens"),
        output_tokens=_get("output_tokens"),
        cache_creation_tokens=_get("cache_creation_input_tokens"),
        cache_read_tokens=_get("cache_read_input_tokens"),
    )


def no_cache_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """The **counterfactual** cost of one call had prompt caching not been in effect.

    The same prompt tokens are repriced at the **full input rate**: every cached-read and
    cache-write token is billed as if it were a fresh ``input_tokens`` token. This is the
    apples-to-apples no-cache baseline for the savings figure (spec section 16) — it reprices
    the *observed* token counts rather than re-running the loop, so there is zero run-to-run
    noise. (The wired provider is Azure OpenAI GPT-5.5, whose prefix caching is **automatic**
    and cannot be disabled via a request parameter; repricing the usage is the honest, exact way
    to measure the caching win. See ``observability/cost.py`` and the Split 09 session log.)
    """
    if model not in PRICES:
        raise ValueError(f"unknown model for pricing: {model!r}")
    input_price, output_price = PRICES[model]
    in_rate = input_price / _PER_MTOK
    out_rate = output_price / _PER_MTOK
    full_prompt = input_tokens + cache_creation_tokens + cache_read_tokens
    return full_prompt * in_rate + output_tokens * out_rate


def savings_vs_no_cache(rows: Any) -> dict[str, float]:
    """Aggregate the with-cache vs no-cache cost of many model calls and the % saved.

    ``rows`` is any iterable of per-call token buckets — each item a mapping (or object) with
    ``model``, ``input_tokens``, ``output_tokens``, ``cache_creation_tokens`` and
    ``cache_read_tokens`` (missing fields → 0). Rows whose ``model`` is ``None``/empty or not in
    :data:`PRICES` (e.g. local ``$0`` RAG tool rows) contribute nothing and are skipped.

    Returns ``{with_cache, no_cache, pct_saved, cache_read_tokens, cache_read_share}`` where
    ``pct_saved`` is the fraction (0–1) by which caching reduced the model spend and
    ``cache_read_share`` is the fraction of prompt tokens served from cache. Pure / deterministic.
    """

    def _get(row: Any, name: str) -> Any:
        return row.get(name) if isinstance(row, dict) else getattr(row, name, None)

    with_cache = 0.0
    no_cache = 0.0
    cache_read = 0
    prompt_tokens = 0
    for row in rows:
        model = _get(row, "model")
        if not model or model not in PRICES:
            continue  # local tool rows / unknown models cost $0 and have no cache buckets
        it = int(_get(row, "input_tokens") or 0)
        ot = int(_get(row, "output_tokens") or 0)
        cc = int(_get(row, "cache_creation_tokens") or 0)
        cr = int(_get(row, "cache_read_tokens") or 0)
        with_cache += cost_usd(model, it, ot, cc, cr)
        no_cache += no_cache_cost_usd(model, it, ot, cc, cr)
        cache_read += cr
        prompt_tokens += it + cc + cr
    pct = (no_cache - with_cache) / no_cache if no_cache > 0 else 0.0
    share = cache_read / prompt_tokens if prompt_tokens > 0 else 0.0
    return {
        "with_cache": with_cache,
        "no_cache": no_cache,
        "pct_saved": pct,
        "cache_read_tokens": cache_read,
        "cache_read_share": share,
    }

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

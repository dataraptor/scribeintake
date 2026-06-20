"""Deterministic tests for the no-cache baseline + caching-savings math (Split 09, §16).

The headline cost number — "$/session with vs without cache, N% saved" — must be *measured*, not
asserted. These tests pin the arithmetic on fixture usage objects: a cache-read row costs strictly
less than the same tokens at full input price, and the % saved is computed correctly. Local ($0)
rows contribute nothing.
"""

import pytest

from scribeintake import pricing


def test_no_cache_reprices_all_prompt_tokens_at_full_input():
    # 1000 fresh + 2000 cache-write + 4000 cache-read prompt tokens on Sonnet ($3/MTok input).
    # No-cache counterfactual bills all 7000 as input: 7000/1e6*3 + output.
    nc = pricing.no_cache_cost_usd(
        "claude-sonnet-4-6",
        input_tokens=1000,
        output_tokens=500,
        cache_creation_tokens=2000,
        cache_read_tokens=4000,
    )
    expected = (1000 + 2000 + 4000) / 1e6 * 3 + 500 / 1e6 * 15
    assert nc == pytest.approx(expected)


def test_cache_read_row_costs_strictly_less_than_full_price():
    """The §16 honesty guard: caching must actually reduce cost for a cache-read row."""
    with_cache = pricing.cost_usd(
        "gpt-5.5", input_tokens=300, output_tokens=40, cache_read_tokens=1800
    )
    no_cache = pricing.no_cache_cost_usd(
        "gpt-5.5", input_tokens=300, output_tokens=40, cache_read_tokens=1800
    )
    assert with_cache < no_cache
    # A row with NO cache reads is priced identically either way (nothing to save).
    plain_with = pricing.cost_usd("gpt-5.5", 300, 40)
    plain_no = pricing.no_cache_cost_usd("gpt-5.5", 300, 40)
    assert plain_with == pytest.approx(plain_no)


def test_savings_vs_no_cache_pct_and_share():
    rows = [
        # cold turn: no cache reads
        {"model": "gpt-5.5", "input_tokens": 1800, "output_tokens": 40, "cache_read_tokens": 0,
         "cache_creation_tokens": 0},
        # warm turn: most of the prefix served from cache
        {"model": "gpt-5.5", "input_tokens": 260, "output_tokens": 36, "cache_read_tokens": 1792,
         "cache_creation_tokens": 0},
    ]
    s = pricing.savings_vs_no_cache(rows)
    # Recompute expected from the same helpers.
    wc = sum(
        pricing.cost_usd(r["model"], r["input_tokens"], r["output_tokens"],
                         r["cache_creation_tokens"], r["cache_read_tokens"])
        for r in rows
    )
    nc = sum(
        pricing.no_cache_cost_usd(r["model"], r["input_tokens"], r["output_tokens"],
                                  r["cache_creation_tokens"], r["cache_read_tokens"])
        for r in rows
    )
    assert s["with_cache"] == pytest.approx(wc)
    assert s["no_cache"] == pytest.approx(nc)
    assert s["pct_saved"] == pytest.approx((nc - wc) / nc)
    assert s["pct_saved"] > 0  # caching saved something
    assert s["cache_read_tokens"] == 1792
    prompt = 1800 + 260 + 1792
    assert s["cache_read_share"] == pytest.approx(1792 / prompt)


def test_savings_skips_local_and_unknown_model_rows():
    rows = [
        {"model": None, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
         "cache_creation_tokens": 0},  # local RAG row → $0
        {"model": "some-unpriced-model", "input_tokens": 999, "output_tokens": 999,
         "cache_read_tokens": 0, "cache_creation_tokens": 0},  # unknown → skipped
        {"model": "gpt-5.5", "input_tokens": 200, "output_tokens": 20, "cache_read_tokens": 1000,
         "cache_creation_tokens": 0},
    ]
    s = pricing.savings_vs_no_cache(rows)
    only = pricing.cost_usd("gpt-5.5", 200, 20, 0, 1000)
    assert s["with_cache"] == pytest.approx(only)


def test_savings_empty_rows_is_zero_not_nan():
    s = pricing.savings_vs_no_cache([])
    assert s == {
        "with_cache": 0.0,
        "no_cache": 0.0,
        "pct_saved": 0.0,
        "cache_read_tokens": 0,
        "cache_read_share": 0.0,
    }


def test_no_cache_unknown_model_raises():
    with pytest.raises(ValueError):
        pricing.no_cache_cost_usd("gpt-4o", 1, 1)

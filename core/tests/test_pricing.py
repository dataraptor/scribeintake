"""Deterministic tests for cache-aware cost accounting (exact arithmetic)."""

import pytest

from scribeintake import pricing


def test_sonnet_basic_input_output():
    # 1000 input + 500 output on Sonnet 4.6 ($3 / $15 per MTok).
    cost = pricing.cost_usd("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)
    expected = 1000 / 1e6 * 3 + 500 / 1e6 * 15
    assert cost == pytest.approx(expected)


def test_opus_and_haiku_prices():
    assert pricing.cost_usd("claude-opus-4-8", 1_000_000, 0) == pytest.approx(5.0)
    assert pricing.cost_usd("claude-opus-4-8", 0, 1_000_000) == pytest.approx(25.0)
    assert pricing.cost_usd("claude-haiku-4-5", 1_000_000, 0) == pytest.approx(1.0)
    assert pricing.cost_usd("claude-haiku-4-5", 0, 1_000_000) == pytest.approx(5.0)


def test_cache_creation_bucket_priced_at_1_25x_input():
    # 1M cache-write tokens on Sonnet ($3/MTok input) => 3 * 1.25 = 3.75.
    cost = pricing.cost_usd("claude-sonnet-4-6", 0, 0, cache_creation_tokens=1_000_000)
    assert cost == pytest.approx(3.0 * 1.25)


def test_cache_read_bucket_priced_at_0_1x_input():
    # 1M cache-read tokens on Sonnet ($3/MTok input) => 3 * 0.1 = 0.30.
    cost = pricing.cost_usd("claude-sonnet-4-6", 0, 0, cache_read_tokens=1_000_000)
    assert cost == pytest.approx(3.0 * 0.10)


def test_all_four_buckets_combined():
    cost = pricing.cost_usd(
        "claude-opus-4-8",
        input_tokens=1000,
        output_tokens=500,
        cache_creation_tokens=2000,
        cache_read_tokens=4000,
    )
    expected = (
        1000 / 1e6 * 5
        + 2000 / 1e6 * 5 * 1.25
        + 4000 / 1e6 * 5 * 0.10
        + 500 / 1e6 * 25
    )
    assert cost == pytest.approx(expected)


def test_cost_from_usage_object_and_dict():
    usage_dict = {
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_creation_input_tokens": 2000,
        "cache_read_input_tokens": 4000,
    }
    from_dict = pricing.cost_from_usage("claude-opus-4-8", usage_dict)
    direct = pricing.cost_usd("claude-opus-4-8", 1000, 500, 2000, 4000)
    assert from_dict == pytest.approx(direct)

    class _Usage:
        input_tokens = 1000
        output_tokens = 500
        cache_creation_input_tokens = 2000
        cache_read_input_tokens = 4000

    from_obj = pricing.cost_from_usage("claude-opus-4-8", _Usage())
    assert from_obj == pytest.approx(direct)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        pricing.cost_usd("gpt-4o", 1, 1)

"""Deterministic wiring test for the cold-vs-warm cache check (Split 09, §7) — no API key.

A scripted structured client reports ``cache_read = 0`` on the first (cold) call and a non-zero
read on warm repeats; the check must surface that as a demonstrated cache hit. The live assertion
against the real model lives in ``test_cache_check_live.py``.
"""

from observability.cache_check import OPENAI_MIN_CACHEABLE, format_result, run_cache_check
from scribeintake.llm import STOP_END_TURN, LLMUsage, StructuredResponse
from scribeintake.models import SOAP, Subjective


class _CachingFakeClient:
    """Returns a valid SOAP each parse; cache_read is 0 on call 1, then ``warm`` on warm calls."""

    model = "gpt-5.5"

    def __init__(self, warm: int = 1280) -> None:
        self.warm = warm
        self.calls = 0

    def parse(self, *, system, messages, schema, effort="high", max_tokens=2048):
        self.calls += 1
        cold = self.calls == 1
        usage = LLMUsage(
            input_tokens=1400 if cold else 128,
            output_tokens=700,
            cache_read_tokens=0 if cold else self.warm,
        )
        return StructuredResponse(
            parsed=SOAP(subjective=Subjective(chief_complaint="sore throat")),
            refused=False,
            stop_reason=STOP_END_TURN,
            usage=usage,
            model=self.model,
        )


def test_cache_check_surfaces_cold_to_warm_read():
    res = run_cache_check(summary_client=_CachingFakeClient(warm=1280), n_calls=3)
    assert res.n_calls == 3
    assert res.cold_cache_read == 0
    assert res.warm_cache_read_max == 1280
    assert res.cache_read_demonstrated is True
    assert res.clears_openai_min is True  # 1280 >= 1024
    assert res.clears_sonnet_floor is False  # 1280 < 2048 (honest: GPT-5.5 prefix, not Claude's)
    assert len(res.per_call) == 3
    assert res.per_call[0]["cache_read_tokens"] == 0
    assert res.per_call[1]["cache_read_tokens"] == 1280


def test_cache_check_no_caching_is_not_demonstrated():
    res = run_cache_check(summary_client=_CachingFakeClient(warm=0), n_calls=2)
    assert res.cache_read_demonstrated is False
    assert res.warm_cache_read_max == 0


def test_format_result_is_ascii_safe():
    res = run_cache_check(summary_client=_CachingFakeClient(), n_calls=2)
    text = format_result(res)
    text.encode("cp1252")  # must not raise — console output stays ASCII
    assert "caching demonstrated  : True" in text
    assert str(OPENAI_MIN_CACHEABLE) in text

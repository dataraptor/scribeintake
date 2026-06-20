"""Cold-vs-warm prompt-cache verification (Split 09, spec section 7) — **live** (needs a key).

Demonstrates prompt caching on the real model by issuing the terminal structured-output call
(``build_summary``) **back-to-back over a fixed completed intake state**: the large, byte-stable
``system + SOAP json_schema`` prefix is cache-cold on the first call and **read** on warm repeats
(``cache_read_tokens > 0``). The unverified claim "we use prompt caching" is a credibility risk —
this turns it into a measured fact.

**Why the terminal call, not the agent loop?** On the wired Azure GPT-5.5 deployment the agent
loop's smaller ``system + tools`` prefix does **not** surface cache hits (verified: cached_tokens
stays 0 across a multi-turn intake), but the terminal call's schema-dominated prefix reliably
caches ~1280 tokens on warm repeats (verified directly) — the honest, reproducible signal.
Provider note: Azure GPT-5.5 has automatic prefix caching with no cache-write bucket, so the
observable signal is ``cache_read`` rising from 0 (cold) to >0 (warm); OpenAI's minimum cacheable
prefix is ~1024 tokens (the spec's Sonnet 2048 floor is a Claude-specific figure, reported for the
§7 mapping but not applicable to this provider).
"""

from __future__ import annotations

from scribeintake.config import CACHE_FLOOR_SONNET, settings
from scribeintake.models import Confidence, IntakeState, SlotValue
from scribeintake.tools.build_summary import build_summary

from .models import CacheCheckResult

# A fixed, completed intake state (the 5 required slots) so the terminal call has stable inputs.
FIXED_SLOTS = [
    ("chief_complaint", "sore throat"),
    ("hpi.onset", "2 days ago"),
    ("hpi.severity", "mild, 3 out of 10"),
    ("medications", "none"),
    ("allergies", "none"),
]

# OpenAI's documented minimum cacheable prefix length (informational threshold).
OPENAI_MIN_CACHEABLE = 1024


def _fixed_state() -> IntakeState:
    state = IntakeState(session_id="cache-check")
    for key, value in FIXED_SLOTS:
        state.slots[key] = SlotValue(value=value, confidence=Confidence.high)
    return state


def run_cache_check(
    *,
    summary_client: object | None = None,
    n_calls: int = 3,
) -> CacheCheckResult:
    """Repeat the terminal structured-output call ``n_calls`` times and measure cold→warm cache
    reads. Builds the live summary client if ``summary_client`` is ``None`` (the live path); tests
    inject a fake whose 2nd+ calls report ``cache_read > 0``.
    """
    if summary_client is None:
        from scribeintake.llm import build_summary_client

        summary_client = build_summary_client(settings)

    state = _fixed_state()
    per_call: list[dict] = []
    for i in range(1, n_calls + 1):
        res = build_summary(
            state, client=summary_client, generated_at="2026-06-20T00:00:00Z"  # type: ignore[arg-type]
        )
        u = res.usage
        per_call.append(
            {"call": i, "input_tokens": u.input_tokens, "cache_read_tokens": u.cache_read_tokens}
        )

    cold = per_call[0]["cache_read_tokens"] if per_call else 0
    warm = per_call[1:]
    warm_max = max((c["cache_read_tokens"] for c in warm), default=0)
    note = (
        "Azure GPT-5.5 automatic prefix caching on the terminal build_summary call: no cache-write "
        f"bucket, so the signal is cache_read rising from {cold} (cold) to {warm_max} (warm). "
        f"Cached prefix = system + SOAP json_schema. OpenAI min cacheable ~{OPENAI_MIN_CACHEABLE} "
        f"tok; spec Sonnet floor {CACHE_FLOOR_SONNET} tok (Claude-specific, informational here)."
    )
    return CacheCheckResult(
        model=settings.ACTIVE_INTAKE_MODEL,
        n_calls=n_calls,
        cold_cache_read=cold,
        warm_cache_read_max=warm_max,
        cache_read_demonstrated=warm_max > 0,
        cached_prefix_tokens=warm_max,
        openai_min_cacheable=OPENAI_MIN_CACHEABLE,
        clears_openai_min=warm_max >= OPENAI_MIN_CACHEABLE,
        sonnet_floor=CACHE_FLOOR_SONNET,
        clears_sonnet_floor=warm_max >= CACHE_FLOOR_SONNET,
        per_call=per_call,
        note=note,
    )


def format_result(res: CacheCheckResult) -> str:
    """One-block human summary (ASCII only — safe for Windows cp1252 stdout)."""
    lines = [
        f"Prompt-cache check  model={res.model}  calls={res.n_calls}",
        f"  cold cache_read       : {res.cold_cache_read}",
        f"  warm cache_read (max) : {res.warm_cache_read_max}",
        f"  caching demonstrated  : {res.cache_read_demonstrated}",
        f"  cached prefix tokens  : {res.cached_prefix_tokens} "
        f"(clears OpenAI ~{res.openai_min_cacheable} min: {res.clears_openai_min}; "
        f"Sonnet {res.sonnet_floor} floor: {res.clears_sonnet_floor})",
        "  per-call cache_read   : "
        + ", ".join(f"c{c['call']}={c['cache_read_tokens']}" for c in res.per_call),
    ]
    return "\n".join(lines)

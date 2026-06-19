"""Test doubles for the LLM provider seam (no network, no key).

A :class:`FakeLLMClient` returns scripted :class:`~scribeintake.llm.LLMResponse` objects in
order and records every call, so the orchestration logic is covered deterministically. The
builder helpers construct the common response shapes (a tool call, a plain question, a
refusal, a truncation).
"""

from __future__ import annotations

from scribeintake.llm import (
    STOP_END_TURN,
    STOP_MAX_TOKENS,
    STOP_REFUSAL,
    STOP_TOOL_USE,
    LLMResponse,
    LLMUsage,
    ToolCall,
)

FAKE_MODEL = "gpt-5.5"  # must be a key in pricing.PRICES so trace cost is non-zero


class FakeLLMClient:
    """Returns queued responses; records calls for assertions."""

    def __init__(self, responses: list[LLMResponse] | None = None, model: str = FAKE_MODEL) -> None:
        self._responses = list(responses or [])
        self.model = model
        self.calls: list[dict] = []

    def complete(self, *, system, messages, tools, max_tokens, effort="medium") -> LLMResponse:
        self.calls.append(
            {
                "system": system,
                "messages": messages,
                "tools": tools,
                "max_tokens": max_tokens,
                "effort": effort,
            }
        )
        if self._responses:
            resp = self._responses.pop(0)
        else:
            # Default tail: a benign plain-text question so a loop never starves.
            resp = text_response("Is there anything else you'd like to add?", model=self.model)
        if not resp.model:
            resp.model = self.model
        return resp


def text_response(
    text: str,
    *,
    model: str = FAKE_MODEL,
    input_tokens: int = 120,
    output_tokens: int = 20,
    cache_read_tokens: int = 0,
) -> LLMResponse:
    return LLMResponse(
        text=text,
        stop_reason=STOP_END_TURN,
        usage=LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
        ),
        model=model,
    )


def tool_response(
    calls: list[tuple],
    *,
    model: str = FAKE_MODEL,
    input_tokens: int = 140,
    output_tokens: int = 30,
) -> LLMResponse:
    """``calls`` is a list of ``(name, args)`` or ``(id, name, args)`` tuples."""
    tcs: list[ToolCall] = []
    for i, c in enumerate(calls):
        if len(c) == 2:
            name, args = c
            cid = f"call_{i}"
        else:
            cid, name, args = c
        tcs.append(ToolCall(id=cid, name=name, arguments=args))
    return LLMResponse(
        text="",
        tool_calls=tcs,
        stop_reason=STOP_TOOL_USE,
        usage=LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        model=model,
    )


def refusal_response(*, model: str = FAKE_MODEL) -> LLMResponse:
    return LLMResponse(
        text="I'm not able to help with that.",
        stop_reason=STOP_REFUSAL,
        usage=LLMUsage(input_tokens=80, output_tokens=5),
        model=model,
    )


def max_tokens_response(*, model: str = FAKE_MODEL) -> LLMResponse:
    return LLMResponse(
        text="",
        stop_reason=STOP_MAX_TOKENS,
        usage=LLMUsage(input_tokens=80, output_tokens=512),
        model=model,
    )

"""Test doubles for the LLM provider seam (no network, no key).

A :class:`FakeLLMClient` returns scripted :class:`~scribeintake.llm.LLMResponse` objects in
order and records every call, so the orchestration logic is covered deterministically. The
builder helpers construct the common response shapes (a tool call, a plain question, a
refusal, a truncation).
"""

from __future__ import annotations

from pydantic import BaseModel

from scribeintake.llm import (
    STOP_END_TURN,
    STOP_MAX_TOKENS,
    STOP_REFUSAL,
    STOP_TOOL_USE,
    LLMResponse,
    LLMUsage,
    StructuredResponse,
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


class FakeStructuredClient:
    """Scripted :class:`~scribeintake.llm.StructuredClient` for the terminal calls (no network).

    ``by_schema`` maps a Pydantic schema's class name (``"SOAP"``, ``"TriageSuggestion"``) to the
    instance ``parse`` should return for that schema. ``refuse`` forces a refusal; ``stop_reasons``
    scripts per-call ``stop_reason``s (e.g. ``["max_tokens", "end_turn"]`` to exercise the
    one-shot backstop). Records every call for assertions.
    """

    def __init__(
        self,
        by_schema: dict[str, BaseModel],
        *,
        model: str = FAKE_MODEL,
        usage: LLMUsage | None = None,
        refuse: bool = False,
        stop_reasons: list[str] | None = None,
    ) -> None:
        self.by_schema = by_schema
        self.model = model
        self._usage = usage or LLMUsage(input_tokens=300, output_tokens=400)
        self._refuse = refuse
        self._stop_reasons = list(stop_reasons or [])
        self.calls: list[dict] = []

    def parse(
        self, *, system, messages, schema, effort="high", max_tokens=2048
    ) -> StructuredResponse:
        self.calls.append(
            {"system": system, "messages": messages, "schema": schema.__name__,
             "effort": effort, "max_tokens": max_tokens}
        )
        stop = self._stop_reasons.pop(0) if self._stop_reasons else STOP_END_TURN
        if self._refuse:
            return StructuredResponse(
                parsed=None, refused=True, stop_reason=STOP_REFUSAL,
                usage=self._usage, model=self.model,
            )
        parsed = self.by_schema[schema.__name__]
        return StructuredResponse(
            parsed=parsed, refused=False, stop_reason=stop, usage=self._usage, model=self.model,
        )

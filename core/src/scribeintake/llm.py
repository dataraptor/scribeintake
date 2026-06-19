"""LLM provider seam — the single place the agent loop talks to a model (spec section 7).

Split 03 wires the agent on **Azure OpenAI GPT-5.5** (the key shipped in ``.env``), not the
spec's Claude pin. To keep that swap clean and the deterministic test tier keyless, the loop
depends only on the small, provider-neutral :class:`LLMClient` protocol below — never on a
concrete SDK. :class:`AzureOpenAIClient` is the live implementation; tests inject a fake that
returns scripted :class:`LLMResponse` objects.

**No sampling knobs.** GPT-5.5 is a reasoning model and rejects ``temperature``/``top_p``/
``seed`` (400) — the same constraint the spec's API-conformance table records for Opus. The
client sends only ``reasoning_effort`` (the route), ``tools``, ``tool_choice`` and
``max_completion_tokens``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

# stop-reason vocabulary (provider-neutral, modelled on Anthropic's ``stop_reason``)
STOP_END_TURN = "end_turn"
STOP_TOOL_USE = "tool_use"
STOP_REFUSAL = "refusal"
STOP_MAX_TOKENS = "max_tokens"


@dataclass
class LLMUsage:
    """Token counts for one model call, normalised to the four cost buckets (pricing.py)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0  # OpenAI has no explicit cache-write bucket -> 0


@dataclass
class ToolCall:
    """One tool invocation the model asked for, with arguments already JSON-decoded."""

    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Normalised result of one model call (one ``chat.completions.create``)."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = STOP_END_TURN
    usage: LLMUsage = field(default_factory=LLMUsage)
    model: str = ""
    # Provider-native assistant message to append to the running history before the
    # tool-result messages (so the next model call sees its own tool_calls turn).
    raw_message: dict | None = None


class LLMClient(Protocol):
    """The only surface the agent loop depends on. ``model`` is recorded in traces."""

    model: str

    def complete(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        effort: str = "medium",
    ) -> LLMResponse:
        """Run one model call. ``messages``/``tools`` are OpenAI chat-completions shapes."""
        ...


# ------------------------------------------------------- structured output (terminal calls)
M = TypeVar("M", bound=BaseModel)


@dataclass
class StructuredResponse:
    """Result of one **native structured-output** call (``build_summary``/``suggest_triage``).

    ``parsed`` is a validated Pydantic instance of the requested schema, or ``None`` on a
    refusal (check ``refused`` before reading ``parsed``). Schema validity is guaranteed at the
    API layer (the model is forced to emit JSON matching the schema), so ``parsed`` never needs
    a generate-then-validate-retry loop.
    """

    parsed: BaseModel | None
    refused: bool
    stop_reason: str
    usage: LLMUsage
    model: str


class StructuredClient(Protocol):
    """Seam for the orchestrator-invoked terminal calls. ``model`` is recorded in traces."""

    model: str

    def parse(
        self,
        *,
        system: str,
        messages: list[dict],
        schema: type[M],
        effort: str = "high",
        max_tokens: int = 2048,
    ) -> StructuredResponse:
        """Run one structured-output call returning a validated instance of ``schema``."""
        ...


# ---------------------------------------------------------------- Azure OpenAI (live)
_FINISH_MAP = {
    "tool_calls": STOP_TOOL_USE,
    "stop": STOP_END_TURN,
    "length": STOP_MAX_TOKENS,
    "content_filter": STOP_REFUSAL,
}


def _assistant_message(text: str, tool_calls: list[ToolCall]) -> dict:
    """Build an OpenAI assistant message (used when a provider can't give a raw one)."""
    msg: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in tool_calls
        ]
    return msg


class AzureOpenAIClient:
    """Live :class:`LLMClient` backed by Azure OpenAI GPT-5.5.

    The ``openai`` SDK is imported lazily so importing this module (and the deterministic
    test tier) never requires the package or any credentials.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        api_version: str,
        model: str,
    ) -> None:
        from openai import AzureOpenAI  # lazy: keeps import-time + keyless tests clean

        self.model = model
        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    def complete(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        effort: str = "medium",
    ) -> LLMResponse:
        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_completion_tokens": max_tokens,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools
            kwargs["tool_choice"] = "auto"
        # GPT-5.5 on /v1/chat/completions rejects ``reasoning_effort`` *together with*
        # function tools (400 — it directs callers to /v1/responses). The intake loop always
        # has tools, so the effort route is only sent on tool-less calls. Migrating the loop
        # to the Responses API to regain per-route effort is deferred (Split 09 territory).
        if effort and not oai_tools:
            kwargs["reasoning_effort"] = effort
        # Deliberately NO temperature/top_p/seed — rejected (400) by this reasoning model.

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        # Refusal: GPT-5 surfaces a structured ``refusal`` field; also map content_filter.
        refusal = getattr(msg, "refusal", None)
        stop = _FINISH_MAP.get(choice.finish_reason, STOP_END_TURN)
        if refusal:
            stop = STOP_REFUSAL

        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = _usage_from_openai(resp.usage)
        text = refusal or msg.content or ""
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop,
            usage=usage,
            model=self.model,
            raw_message=_assistant_message(msg.content or "", tool_calls),
        )

    def parse(
        self,
        *,
        system: str,
        messages: list[dict],
        schema: type[M],
        effort: str = "high",
        max_tokens: int = 2048,
    ) -> StructuredResponse:
        """Native structured output via ``beta.chat.completions.parse`` (no tools).

        The SDK compiles ``schema`` into a strict ``json_schema`` response format and validates
        the model's JSON into the Pydantic instance, guaranteeing schema validity at the API
        layer. Unlike the tool-using loop, this call has **no tools**, so ``reasoning_effort``
        is accepted here (verified 2026-06-20) — restoring per-route effort for the terminal
        summary/triage calls. Deliberately NO temperature/top_p/seed (rejected, 400).
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "response_format": schema,
            "max_completion_tokens": max_tokens,
        }
        if effort:
            kwargs["reasoning_effort"] = effort

        resp = self._client.beta.chat.completions.parse(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        refusal = getattr(msg, "refusal", None)
        stop = _FINISH_MAP.get(choice.finish_reason, STOP_END_TURN)
        if refusal:
            stop = STOP_REFUSAL
        parsed = None if refusal else msg.parsed
        return StructuredResponse(
            parsed=parsed,
            refused=bool(refusal),
            stop_reason=stop,
            usage=_usage_from_openai(resp.usage),
            model=self.model,
        )


def _usage_from_openai(usage: Any) -> LLMUsage:
    """Map an OpenAI ``CompletionUsage`` (or dict) to the neutral :class:`LLMUsage`."""
    if usage is None:
        return LLMUsage()

    def _get(obj: Any, name: str) -> int:
        val = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, 0)
        return int(val or 0)

    details = (
        usage.get("prompt_tokens_details")
        if isinstance(usage, dict)
        else getattr(usage, "prompt_tokens_details", None)
    )
    cached = _get(details, "cached_tokens") if details is not None else 0
    prompt = _get(usage, "prompt_tokens")
    return LLMUsage(
        input_tokens=max(prompt - cached, 0),  # cost buckets must not double-count cache
        output_tokens=_get(usage, "completion_tokens"),
        cache_read_tokens=cached,
    )


def build_azure_client(settings: Any) -> AzureOpenAIClient:
    """Construct the live client from :data:`scribeintake.config.settings`.

    Raises a clear error (only when invoked) if Azure credentials are absent — the
    deterministic tier never reaches here because it injects a fake client.
    """
    if not settings.azure_openai_endpoint or not settings.azure_openai_api_key:
        raise RuntimeError(
            "Azure OpenAI credentials missing: set AZURE_OPENAI_ENDPOINT and "
            "AZURE_OPENAI_API_KEY (see .env) to run the live agent loop."
        )
    return AzureOpenAIClient(
        endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.openai_api_version,
        model=settings.ACTIVE_INTAKE_MODEL,
    )


def build_summary_client(settings: Any) -> AzureOpenAIClient:
    """Construct the live structured-output client for the terminal summary/triage calls.

    The spec pins Opus 4.8 for these; this environment ships an Azure GPT-5.5 key, so the same
    deployment serves both the loop and the terminal calls (it supports
    ``beta.chat.completions.parse``). Kept as a separate factory so a dedicated summary
    deployment can be swapped in later without touching the orchestrator.
    """
    return build_azure_client(settings)

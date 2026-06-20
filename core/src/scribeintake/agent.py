"""The intake agent: a bounded tool-use loop on the active model (spec section 7).

One :meth:`AgentLoop.run_turn` drives a single patient turn: it calls the model, dispatches any
tools it requests through the :class:`~scribeintake.tools.ToolRegistry`, feeds results back, and
returns when the model emits a plain-text question (or a bound is hit). It is **single-model**
(no mid-loop model switch — that would invalidate the prompt cache) and sends **no** sampling
knobs (``temperature``/``top_p``/``seed``), per the API-conformance rule.

The loop is provider-agnostic: it depends only on :class:`~scribeintake.llm.LLMClient`. Split 03
wires it to Azure OpenAI GPT-5.5 (:func:`build_default_agent`); tests inject a fake client.

What the loop does **not** do: decide the conversation is "done" (the orchestrator owns the
deterministic completion check) and send safety templates (the orchestrator owns escalation).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources

from .config import EFFORT_INTAKE, MAX_AGENT_STEPS, settings
from .llm import (
    STOP_MAX_TOKENS,
    STOP_REFUSAL,
    LLMClient,
    LLMResponse,
    LLMUsage,
    ToolCall,
)
from .tools import ToolContext, ToolRegistry, default_registry

# Loaded once; byte-stable across turns (a changing system prompt invalidates the cache, §7).
SYSTEM_PROMPT = (
    resources.files("scribeintake").joinpath("prompts/system.md").read_text(encoding="utf-8")
)

DEFAULT_MAX_TOKENS = 1024

# Patient-facing labels for the live "what the agent is doing" strip (the SSE ``stage`` events).
# Each routine turn makes ~3 sequential model calls; surfacing the real tool dispatch turns that
# unavoidable wait into a window onto the architecture (gate → reason → tools → reason → reply).
# Keyed by the actual tool name so the demo shows the genuine function the model invoked.
TOOL_STAGE_LABELS = {
    "record_intake": "Recording what you told me",
    "retrieve_guideline": "Consulting clinical guidelines",
    "assess_escalation": "Assessing urgency",
}
_THINKING_LABEL = "Thinking…"

# A no-op sink so the loop never has to null-check the callback.
ProgressFn = Callable[[dict], None]


def _noop(_event: dict) -> None:
    pass

# Safe templated replies (code, not model output) for the failure paths (§17/§18).
REFUSAL_REPLY = (
    "I can't help with that part here, but I can still help prepare your information for a "
    "clinician, who will make any medical decisions. Could you tell me a little more about "
    "what's been going on?"
)
STEP_LIMIT_REPLY = (
    "Thanks for bearing with me. To keep things clear — what is the main symptom that's "
    "concerning you most right now?"
)


@dataclass
class AgentStep:
    """One model call (for the trace): tokens, latency, what it asked for."""

    model: str
    text: str
    tool_calls: list[ToolCall]
    usage: LLMUsage
    latency_ms: int


@dataclass
class ToolExecution:
    """One local tool run (for the trace)."""

    name: str
    arguments: dict
    result: dict
    latency_ms: int


@dataclass
class AgentResult:
    """Outcome of one agent turn."""

    assistant_text: str
    steps: list[AgentStep] = field(default_factory=list)
    tool_executions: list[ToolExecution] = field(default_factory=list)
    refused: bool = False
    hit_step_limit: bool = False

    @property
    def tools_used(self) -> list[str]:
        seen: list[str] = []
        for ex in self.tool_executions:
            if ex.name not in seen:
                seen.append(ex.name)
        return seen


class AgentLoop:
    """Bounded tool-use loop over a :class:`LLMClient`."""

    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry | None = None,
        *,
        max_steps: int = MAX_AGENT_STEPS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.client = client
        self.registry = registry or default_registry()
        self.max_steps = max_steps
        self.max_tokens = max_tokens

    def run_turn(
        self,
        *,
        history: list[dict],
        user_content: list[dict] | str,
        ctx: ToolContext,
        system: str = SYSTEM_PROMPT,
        effort: str = EFFORT_INTAKE,
        on_event: ProgressFn | None = None,
    ) -> AgentResult:
        """Run one patient turn. ``history`` is prior messages (OpenAI shape); the volatile
        per-turn reminder is expected to already live inside ``user_content`` (user turn).

        ``on_event`` (optional) receives a small dict per observable step — one ``thinking``
        event before each model call and one ``tool`` event per tool the model invokes — so a
        streaming caller can show the patient what the agent is doing while it waits. It is a
        best-effort UI signal only; it never changes the turn's outcome.
        """
        emit = on_event or _noop
        messages: list[dict] = [*history, {"role": "user", "content": user_content}]
        tools = self.registry.schemas()
        steps: list[AgentStep] = []
        executions: list[ToolExecution] = []

        for _ in range(self.max_steps):
            emit({"stage": "thinking", "label": _THINKING_LABEL})
            resp, latency = self._complete(system, messages, tools, effort)
            steps.append(
                AgentStep(
                    model=resp.model,
                    text=resp.text,
                    tool_calls=resp.tool_calls,
                    usage=resp.usage,
                    latency_ms=latency,
                )
            )

            if resp.stop_reason == STOP_REFUSAL:
                return AgentResult(
                    assistant_text=REFUSAL_REPLY,
                    steps=steps,
                    tool_executions=executions,
                    refused=True,
                )

            if resp.tool_calls:
                messages.append(resp.raw_message or _assistant_msg(resp))
                emergency = False
                for tc in resp.tool_calls:
                    emit(
                        {
                            "stage": "tool",
                            "tool": tc.name,
                            "label": TOOL_STAGE_LABELS.get(tc.name, tc.name),
                        }
                    )
                    result, latency = self.registry.dispatch(tc.name, tc.arguments, ctx)
                    executions.append(ToolExecution(tc.name, tc.arguments, result, latency))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result),
                        }
                    )
                    if ctx.agent_escalation is not None and _is_emergency(ctx):
                        emergency = True
                if emergency:
                    # Agent declared an emergency — stop asking; the orchestrator takes over.
                    return AgentResult(
                        assistant_text="",
                        steps=steps,
                        tool_executions=executions,
                    )
                continue

            # Plain-text response = the assistant's question for this turn.
            return AgentResult(
                assistant_text=resp.text,
                steps=steps,
                tool_executions=executions,
            )

        # Exceeded the tool-call bound: stop and ask a clarifying question (§7).
        return AgentResult(
            assistant_text=STEP_LIMIT_REPLY,
            steps=steps,
            tool_executions=executions,
            hit_step_limit=True,
        )

    def _complete(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        effort: str,
    ) -> tuple[LLMResponse, int]:
        """One model call with a single max_tokens retry; returns (response, latency_ms)."""
        t0 = time.perf_counter()
        resp = self.client.complete(
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=self.max_tokens,
            effort=effort,
        )
        if resp.stop_reason == STOP_MAX_TOKENS:
            resp = self.client.complete(
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=self.max_tokens * 2,
                effort=effort,
            )
        latency = int((time.perf_counter() - t0) * 1000)
        return resp, latency


def _assistant_msg(resp: LLMResponse) -> dict:
    """Fallback assistant message if a client didn't supply ``raw_message``."""
    msg: dict = {"role": "assistant", "content": resp.text or None}
    if resp.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in resp.tool_calls
        ]
    return msg


def _is_emergency(ctx: ToolContext) -> bool:
    from .models import EscalationLevel

    return ctx.agent_escalation is EscalationLevel.EMERGENCY


def build_default_agent() -> AgentLoop:
    """Construct the live agent (Azure OpenAI GPT-5.5) from :data:`settings`.

    Only called when the orchestrator runs a real turn with no injected agent; the keyless
    deterministic tier always injects a fake, so the SDK/credentials are never needed there.
    """
    from .llm import build_azure_client

    return AgentLoop(build_azure_client(settings))

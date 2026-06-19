"""Tool base types (kept separate from ``__init__`` to avoid an import cycle).

The concrete tool modules import :class:`ToolSpec`/:class:`ToolContext` from here; the
package ``__init__`` then imports those modules and assembles the registry.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import RULES_VERSION
from ..models import EscalationLevel, IntakeState


@dataclass
class ToolContext:
    """Per-turn scratch space the tools read/write; the orchestrator owns the lifecycle.

    ``state`` is the live :class:`IntakeState` (tools mutate ``state.slots`` in place); the
    orchestrator persists it after the loop. ``agent_escalation`` is the independent net's
    verdict (escalate-only); the orchestrator acts on it (template + halt for EMERGENCY).
    """

    session_id: str
    turn: int
    state: IntakeState
    conn: object | None = None  # sqlite3.Connection | None (kept loose to avoid import)
    rules_version: str = RULES_VERSION
    msg_id: str | None = None  # the current patient message id (slot-write provenance)
    open_slots: list[str] = field(default_factory=list)
    branch_hints: list[str] = field(default_factory=list)
    agent_escalation: EscalationLevel | None = None
    agent_escalation_rationale: str = ""
    # Live RAG retriever for ``retrieve_guideline`` (Split 05). Injected by the orchestrator;
    # ``None`` ⇒ the tool returns no chunks (graceful "no citation available"), so the
    # deterministic tier needs no built index.
    retriever: object | None = None  # rag.HybridRetriever | None


@dataclass
class ToolSpec:
    """A registered tool: JSON schema for the model + a local executor."""

    name: str
    description: str
    parameters: dict
    executor: Callable[[dict, ToolContext], dict]


class ToolRegistry:
    """Holds tool specs; renders the model-facing list and dispatches calls."""

    def __init__(self, specs: list[ToolSpec]) -> None:
        self._by_name = {s.name: s for s in specs}

    def schemas(self) -> list[dict]:
        """Provider-neutral tool list ({name, description, parameters})."""
        return [
            {"name": s.name, "description": s.description, "parameters": s.parameters}
            for s in self._by_name.values()
        ]

    def names(self) -> list[str]:
        return list(self._by_name)

    def dispatch(self, name: str, arguments: dict, ctx: ToolContext) -> tuple[dict, int]:
        """Run one tool; returns ``(result, latency_ms)``.

        An unknown tool name yields a structured error (the model occasionally hallucinates
        a name); a tool that raises is reported as an error result rather than crashing the
        turn. Safety escalation is never swallowed: it flows through ``ctx.agent_escalation``
        which the escalation tool sets before any failure point.
        """
        spec = self._by_name.get(name)
        t0 = time.perf_counter()
        if spec is None:
            return {"error": f"unknown tool: {name}"}, _ms(t0)
        try:
            result = spec.executor(arguments or {}, ctx)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the turn
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return result, _ms(t0)


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)

"""Agent tool registry (spec section 8).

The three **agent-chosen** tools — ``record_intake``, ``retrieve_guideline`` (a stub this
split) and ``assess_escalation`` — plus the per-turn :class:`ToolContext` they mutate and
the :class:`ToolRegistry` the agent loop dispatches through. ``build_summary`` and
``suggest_triage`` are deliberately **not** here: they are orchestrator-invoked terminal
calls (Split 04), never agent-chosen.
"""

from __future__ import annotations

from . import assess_escalation as _assess
from . import record_intake as _record
from . import retrieve_guideline as _retrieve
from .base import ToolContext, ToolRegistry, ToolSpec

__all__ = [
    "ToolContext",
    "ToolSpec",
    "ToolRegistry",
    "default_registry",
]


def default_registry() -> ToolRegistry:
    """The three agent tools, in clinical priority order."""
    return ToolRegistry([_record.SPEC, _retrieve.SPEC, _assess.SPEC])

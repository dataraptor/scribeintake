"""``assess_escalation`` tool (spec section 8) — the agent's independent safety net.

This is a *different method* from the deterministic code gate: the LLM, reading the
conversation, may raise an escalation the regex/number extractor missed (covering the
extraction soft-link). It is **escalate-only — it can never de-escalate**: a ``CLEAR`` is
ignored, and a level no higher than one already recorded this turn is a no-op.

The verdict is recorded on :attr:`ToolContext.agent_escalation`; the **orchestrator** acts on
it (an ``EMERGENCY`` is handled exactly like a gate emergency: template + halt). A non-CLEAR
assessment is logged to ``safety_events`` with ``source="agent"`` here, so the audit reflects
that the agent — not the gate — raised it.
"""

from __future__ import annotations

from ..models import AssessEscalationInput, EscalationLevel
from .base import ToolContext, ToolSpec

_RANK = {
    EscalationLevel.CLEAR: 0,
    EscalationLevel.URGENT: 1,
    EscalationLevel.EMERGENCY: 2,
}

_DESCRIPTION = (
    "Independently assess whether the conversation warrants escalation beyond routine "
    "intake. Use EMERGENCY for life-threatening red flags, URGENT for same-day concerns. "
    "You may only escalate, never downgrade. The system, not you, sends any safety message."
)


def execute(arguments: dict, ctx: ToolContext) -> dict:
    """Record an escalate-only verdict; log a safety_event for non-CLEAR levels."""
    payload = AssessEscalationInput.model_validate(arguments)
    level = payload.level

    # Escalate-only: CLEAR never lowers anything; a level <= the one already recorded is a no-op.
    current = ctx.agent_escalation
    if level is EscalationLevel.CLEAR or (current is not None and _RANK[level] <= _RANK[current]):
        return {"acknowledged": False}

    ctx.agent_escalation = level
    ctx.agent_escalation_rationale = payload.rationale
    _log_safety_event(ctx, level)
    return {"acknowledged": True}


def _log_safety_event(ctx: ToolContext, level: EscalationLevel) -> None:
    """Best-effort safety_events audit row (never raises into the tool/loop)."""
    if ctx.conn is None or not ctx.session_id:
        return
    try:
        from ..db import log_safety_event

        log_safety_event(
            ctx.conn,
            session_id=ctx.session_id,
            level=level.value,
            source="agent",
            matched_rules=["agent_assessment"],
            rules_version=ctx.rules_version,
            msg_id=None,
        )
    except Exception:
        pass


SPEC = ToolSpec(
    name="assess_escalation",
    description=_DESCRIPTION,
    parameters=AssessEscalationInput.model_json_schema(),
    executor=execute,
)

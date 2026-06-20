"""``assess_escalation`` is escalate-only (Split 13 §3.2) — NO API key.

The agent's independent net (Split 03 tool) may **raise** an escalation the regex extractor
missed, but it can **never lower** one: a ``CLEAR`` is ignored, and a level no higher than one
already recorded this turn is a no-op. This closes the "the LLM was talked into downgrading"
attack — the tool's contract, not the model's discretion, decides.
"""

from __future__ import annotations

from scribeintake.models import AssessEscalationInput, EscalationLevel, IntakeState
from scribeintake.tools.assess_escalation import execute as assess_escalation
from scribeintake.tools.base import ToolContext


def _ctx() -> ToolContext:
    return ToolContext(session_id="t", turn=1, state=IntakeState(session_id="t"))


def _call(ctx: ToolContext, level: EscalationLevel) -> dict:
    return assess_escalation(
        AssessEscalationInput(level=level, rationale="test").model_dump(), ctx
    )


def test_clear_is_a_no_op():
    ctx = _ctx()
    result = _call(ctx, EscalationLevel.CLEAR)
    assert result["acknowledged"] is False
    assert ctx.agent_escalation is None


def test_emergency_is_honored():
    ctx = _ctx()
    result = _call(ctx, EscalationLevel.EMERGENCY)
    assert result["acknowledged"] is True
    assert ctx.agent_escalation is EscalationLevel.EMERGENCY


def test_urgent_cannot_downgrade_a_recorded_emergency():
    ctx = _ctx()
    _call(ctx, EscalationLevel.EMERGENCY)
    result = _call(ctx, EscalationLevel.URGENT)
    assert result["acknowledged"] is False
    assert ctx.agent_escalation is EscalationLevel.EMERGENCY  # the floor held


def test_clear_cannot_downgrade_a_recorded_urgent():
    ctx = _ctx()
    _call(ctx, EscalationLevel.URGENT)
    result = _call(ctx, EscalationLevel.CLEAR)
    assert result["acknowledged"] is False
    assert ctx.agent_escalation is EscalationLevel.URGENT


def test_urgent_then_emergency_escalates():
    """The only allowed direction is *up* — URGENT then EMERGENCY records EMERGENCY."""
    ctx = _ctx()
    _call(ctx, EscalationLevel.URGENT)
    result = _call(ctx, EscalationLevel.EMERGENCY)
    assert result["acknowledged"] is True
    assert ctx.agent_escalation is EscalationLevel.EMERGENCY

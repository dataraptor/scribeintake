"""Unit tests for the bounded tool-use loop (LLM mocked, no key/network)."""

from __future__ import annotations

from fakes import (
    FakeLLMClient,
    max_tokens_response,
    refusal_response,
    text_response,
    tool_response,
)

from scribeintake.agent import REFUSAL_REPLY, STEP_LIMIT_REPLY, AgentLoop
from scribeintake.models import EscalationLevel, IntakeState
from scribeintake.tools import ToolContext, default_registry


def _ctx() -> ToolContext:
    state = IntakeState(session_id="s1")
    return ToolContext(session_id="s1", turn=1, state=state, conn=None)


def _run(client: FakeLLMClient, ctx: ToolContext, **kw):
    loop = AgentLoop(client, default_registry(), **kw)
    return loop.run_turn(history=[], user_content="I have a sore throat.", ctx=ctx)


def test_tool_then_question_dispatches_and_records_slots():
    record = tool_response(
        [("record_intake", {"updates": [{"slot": "chief_complaint", "value": "sore throat"}]})]
    )
    client = FakeLLMClient([record, text_response("How many days has it lasted?")])
    ctx = _ctx()
    result = _run(client, ctx)

    assert result.assistant_text == "How many days has it lasted?"
    assert result.tools_used == ["record_intake"]
    assert ctx.state.slots["chief_complaint"].value == "sore throat"
    # open_slots recomputed by the tool and chief_complaint no longer open
    assert "chief_complaint" not in ctx.open_slots
    # two model calls + one tool execution recorded
    assert len(result.steps) == 2
    assert len(result.tool_executions) == 1


def test_on_event_reports_thinking_and_tool_stages_in_order():
    record = tool_response(
        [("record_intake", {"updates": [{"slot": "chief_complaint", "value": "sore throat"}]})]
    )
    client = FakeLLMClient([record, text_response("How many days?")])
    events: list[dict] = []
    loop = AgentLoop(client, default_registry())
    loop.run_turn(
        history=[], user_content="I have a sore throat.", ctx=_ctx(), on_event=events.append
    )

    stages = [(e["stage"], e.get("tool")) for e in events]
    # one thinking per model call (2), one tool event for the dispatched record_intake, in order
    assert stages == [
        ("thinking", None),
        ("tool", "record_intake"),
        ("thinking", None),
    ]
    # the tool event carries a patient-facing label
    tool_ev = next(e for e in events if e["stage"] == "tool")
    assert tool_ev["label"] == "Recording what you told me"


def test_on_event_is_optional_and_defaults_to_noop():
    client = FakeLLMClient([text_response("What's bothering you?")])
    # no on_event passed → must not raise
    result = _run(client, _ctx())
    assert result.assistant_text == "What's bothering you?"


def test_text_only_first_response_is_the_question():
    client = FakeLLMClient([text_response("What's bothering you today?")])
    result = _run(client, _ctx())
    assert result.assistant_text == "What's bothering you today?"
    assert result.tool_executions == []
    assert len(result.steps) == 1


def test_refusal_returns_safe_reply_without_crashing():
    client = FakeLLMClient([refusal_response()])
    result = _run(client, _ctx())
    assert result.refused is True
    assert result.assistant_text == REFUSAL_REPLY


def test_max_tokens_retries_once_then_uses_followup():
    # First call truncates; loop retries with doubled budget; second (retry) returns text.
    client = FakeLLMClient([max_tokens_response(), text_response("Tell me more.")])
    result = _run(client, _ctx(), max_tokens=256)
    assert result.assistant_text == "Tell me more."
    # The retry doubled the budget on the second call.
    assert client.calls[0]["max_tokens"] == 256
    assert client.calls[1]["max_tokens"] == 512


def test_step_limit_stops_and_asks_clarifying_question():
    # Every response asks for a tool, so the loop never terminates naturally.
    forever = [
        tool_response([("retrieve_guideline", {"query": "x", "k": 3})]) for _ in range(10)
    ]
    client = FakeLLMClient(forever)
    result = _run(client, _ctx(), max_steps=3)
    assert result.hit_step_limit is True
    assert result.assistant_text == STEP_LIMIT_REPLY
    assert len(result.steps) == 3  # bounded


def test_assess_escalation_emergency_breaks_loop_early():
    # The agent asks a tool and escalates in the same response; loop must stop, not ask on.
    client = FakeLLMClient(
        [
            tool_response(
                [("assess_escalation", {"level": "EMERGENCY", "rationale": "describes ACS"})]
            ),
            text_response("this question should never be reached"),
        ]
    )
    ctx = _ctx()
    result = _run(client, ctx)
    assert ctx.agent_escalation is EscalationLevel.EMERGENCY
    assert result.assistant_text == ""  # pending question discarded by the loop
    assert len(result.steps) == 1  # broke after the escalation step


def test_no_sampling_knobs_sent():
    client = FakeLLMClient([text_response("ok")])
    _run(client, _ctx())
    call = client.calls[0]
    assert "effort" in call  # reasoning route is sent
    # The neutral client interface has no temperature/top_p/seed params at all.
    assert set(call) == {"system", "messages", "tools", "max_tokens", "effort"}

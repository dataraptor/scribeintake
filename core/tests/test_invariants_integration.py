"""The product thesis, re-asserted in one place — the six safety invariants together.

Split 14 (release acceptance). Every prior split has its own focused tests for these
guarantees; this module is the **single regression guard** a future change must keep green.
If this file is red, the headline claim — *the safety guarantee is **code, upstream of the
LLM***, not the model choosing to behave — is broken.

The six invariants (PROGRESS §2):
  1. ``safety/extractor.py`` contains **no LLM/network call** (it is a pure function).
  2. The safety gate runs in **code, upstream of the agent** (the agent sees a state whose
     floor the gate already pinned).
  3. On a gate **EMERGENCY** the agent is **never invoked** (call-count 0) and the session halts.
  4. The triage floor is **monotonic** across a multi-turn session (a floor never lowers).
  5. A forced exception in the safety path **fails safe** (escalates; never silently CLEAR).
  6. The predicted triage band is **never below** the floor (the clamp), over the full sweep.

The orchestration parts mock the LLM (a scripted ``AgentLoop`` / a ``Mock`` agent), so the whole
file runs in the **deterministic tier** with no API key.
"""

from __future__ import annotations

import inspect
import itertools
from unittest.mock import Mock

import pytest
from fakes import FakeLLMClient, text_response, tool_response

import scribeintake.safety as safety
from scribeintake.agent import AgentLoop
from scribeintake.models import EscalationLevel, EscalationSource, TriageBand
from scribeintake.orchestrator import run_turn
from scribeintake.safety import run_gate
from scribeintake.safety.rules import raise_floor
from scribeintake.tools import default_registry
from scribeintake.tools.suggest_triage import clamp_band

# Tokens that would betray a model/network call living inside a must-be-deterministic path.
_FORBIDDEN_CALLS = (
    "anthropic",
    "openai",
    "messages.create",
    "chat.completions",
    "requests.post",
    "requests.get",
    "httpx.post",
    "httpx.get",
    "urllib",
    "socket",
)


def _recording_agent(*responses) -> tuple[AgentLoop, list]:
    """A real ``AgentLoop`` (scripted client) that records the floor seen at call time.

    Wrapping ``run_turn`` lets invariant 2 prove the gate ran **before** the agent: by the time
    the agent is invoked, the gate has already pinned the floor onto ``ctx.state``.
    """
    loop = AgentLoop(FakeLLMClient(list(responses)), default_registry())
    seen: list[TriageBand] = []
    real_run = loop.run_turn

    def recording_run_turn(*, history, user_content, ctx, effort):
        seen.append(ctx.state.triage_floor)
        return real_run(history=history, user_content=user_content, ctx=ctx, effort=effort)

    loop.run_turn = recording_run_turn  # type: ignore[method-assign]
    return loop, seen


# ----------------------------------------------------------------- invariant 1
def test_invariant_1_extractor_has_no_llm_or_network_call():
    """The extractor (and the rule engine it feeds) is pure code — no model, no I/O."""
    from scribeintake.safety import extractor, rules

    src = inspect.getsource(extractor) + inspect.getsource(rules)
    lowered = src.lower()
    for bad in _FORBIDDEN_CALLS:
        assert bad not in lowered, f"forbidden call token in a deterministic path: {bad!r}"


# ----------------------------------------------------------------- invariant 2
def test_invariant_2_gate_runs_upstream_of_agent(conn, session):
    """The agent sees a state whose floor the gate has already pinned (gate → agent order)."""
    agent, floors_at_call = _recording_agent(
        tool_response([("record_intake", {"updates": [{"slot": "chief_complaint",
                                                        "value": "abdominal pain"}]})]),
        text_response("Where exactly is the pain?"),
    )
    turn = run_turn(
        session,
        "I have severe abdominal pain that is really bad.",
        conn=conn,
        agent=agent,
    )
    # The agent ran (a non-emergency turn), and at that moment the gate-pinned floor was visible.
    assert floors_at_call == [TriageBand.gp_urgent]
    assert turn.triage_floor is TriageBand.gp_urgent


# ----------------------------------------------------------------- invariant 3
def test_invariant_3_emergency_short_circuits_agent_never_runs(conn, session):
    """On a gate EMERGENCY the agent is never invoked and the session halts."""
    agent = Mock()  # any attribute access that runs the agent shows up as a call
    turn = run_turn(
        session,
        "I have crushing chest pain spreading to my left arm and I'm sweating.",
        conn=conn,
        agent=agent,
    )
    assert agent.run_turn.call_count == 0
    assert turn.level is EscalationLevel.EMERGENCY
    assert turn.source is EscalationSource.gate
    assert turn.status == "halted"
    assert turn.template is not None and turn.template["kind"] == "emergency"


# ----------------------------------------------------------------- invariant 4
def test_invariant_4_floor_is_monotonic_across_a_session(conn, session):
    """An EMERGENCY pins ER; a later benign turn cannot lower the floor."""
    agent = Mock()
    t1 = run_turn(session, "I can't breathe and my chest is crushing.", conn=conn, agent=agent)
    assert t1.triage_floor is TriageBand.ER

    # A subsequent (benign-sounding) turn is processed by the gate; the ER floor must hold.
    agent2, _ = _recording_agent(text_response("Can you tell me more?"))
    t2 = run_turn(session, "Actually I feel a bit better now.", conn=conn, agent=agent2)
    assert t2.triage_floor is TriageBand.ER  # never lowered


@pytest.mark.parametrize(
    "sequence,expected",
    [
        ([EscalationLevel.CLEAR], TriageBand.self_care),
        ([EscalationLevel.URGENT, EscalationLevel.CLEAR], TriageBand.gp_urgent),
        ([EscalationLevel.EMERGENCY, EscalationLevel.URGENT, EscalationLevel.CLEAR], TriageBand.ER),
        ([EscalationLevel.URGENT, EscalationLevel.EMERGENCY], TriageBand.ER),
    ],
)
def test_invariant_4_raise_floor_ratchets_up_only(sequence, expected):
    """The pure ``raise_floor`` never lowers, for representative escalation sequences."""
    floor = TriageBand.self_care
    for level in sequence:
        floor = raise_floor(floor, level)
    assert floor is expected


# ----------------------------------------------------------------- invariant 5
def test_invariant_5_safety_path_exception_fails_safe(monkeypatch):
    """A forced exception inside the gate escalates to caution — never a silent CLEAR."""
    def _boom(*_a, **_k):
        raise RuntimeError("forced safety-path failure")

    monkeypatch.setattr(safety, "evaluate", _boom)
    r = run_gate("I have a mild cough")
    assert r.failed_safe is True
    assert r.verdict.level is not EscalationLevel.CLEAR
    assert r.verdict.level is EscalationLevel.URGENT
    assert r.floor is TriageBand.gp_urgent


def test_invariant_5_failsafe_never_lowers_a_pinned_floor(monkeypatch):
    """Failing safe escalates, but must not drop an already-pinned ER floor."""
    def _boom(*_a, **_k):
        raise RuntimeError("forced safety-path failure")

    monkeypatch.setattr(safety, "evaluate", _boom)
    r = run_gate("anything", current_floor=TriageBand.ER)
    assert r.failed_safe is True
    assert r.floor is TriageBand.ER


# ----------------------------------------------------------------- invariant 6
@pytest.mark.parametrize("model_band,floor", list(itertools.product(TriageBand, TriageBand)))
def test_invariant_6_predicted_band_never_below_floor(model_band: TriageBand, floor: TriageBand):
    """The clamp guarantees ``predicted >= floor`` for every (model band, floor) pair."""
    order = list(TriageBand)
    result = clamp_band(model_band, floor)
    assert order.index(result) >= order.index(floor)
    assert result is max(model_band, floor, key=order.index)

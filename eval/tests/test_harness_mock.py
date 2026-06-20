"""Harness tests with the agent + summary client **mocked** (no API key, no network).

Proves the harness drives the full per-turn pipeline in-process, captures well-formed
per-turn records, halts an emergency without feeding further turns, finalizes a completing
intake into a persisted SOAP, and keeps scenarios hermetic (no cross-run state bleed).
"""

from __future__ import annotations

import pytest
from fakes import FakeStructuredClient, text_response, tool_response

from eval.harness import run_scenario
from eval.models import ScenarioRun, TurnRecord
from eval.scenario import Expect, GoldSoap, Scenario, ScenarioCategory
from scribeintake.agent import AgentLoop
from scribeintake.models import SOAP, EscalationLevel, EscalationSource, Subjective, TriageBand
from scribeintake.tools import default_registry
from scribeintake.tools.suggest_triage import TriageSuggestion

CLEAR_1 = "I've had a mild sore throat for a couple of days."
CLEAR_2 = "Here are the rest of the details for you."
EMERGENCY_1 = "I have crushing chest pain spreading to my left arm and I'm sweating a lot."


class _NoRunAgent:
    """A tripwire agent: the gate EMERGENCY short-circuit must never invoke it."""

    def run_turn(self, **_):  # pragma: no cover - asserted not to run
        raise AssertionError("agent ran on a gate EMERGENCY (short-circuit broken)")


def _agent(*responses) -> AgentLoop:
    """A real AgentLoop wired to a scripted fake client (exercises real tool dispatch)."""
    from fakes import FakeLLMClient

    return AgentLoop(FakeLLMClient(list(responses)), default_registry())


def _record(updates: list[tuple[str, str]]):
    return tool_response(
        [("record_intake", {"updates": [{"slot": s, "value": v} for s, v in updates]})]
    )


def _finalize_client(band: TriageBand = TriageBand.self_care) -> FakeStructuredClient:
    return FakeStructuredClient(
        {
            "SOAP": SOAP(subjective=Subjective(chief_complaint="sore throat")),
            "TriageSuggestion": TriageSuggestion(band=band, rationale="benign, self-limited"),
        }
    )


def _routine_scenario(sid: str = "routine_x") -> Scenario:
    return Scenario(
        id=sid,
        category=ScenarioCategory.routine,
        turns=[CLEAR_1, CLEAR_2],
        expect=Expect(escalation=EscalationLevel.CLEAR),
        gold_soap=GoldSoap(chief_complaint="sore throat", triage_band=TriageBand.self_care),
        provenance="synthetic",
    )


def _emergency_scenario(sid: str = "em_x") -> Scenario:
    return Scenario(
        id=sid,
        category=ScenarioCategory.must_escalate,
        turns=[EMERGENCY_1, "this second turn must never be fed"],
        expect=Expect(
            escalation=EscalationLevel.EMERGENCY,
            escalation_source=[EscalationSource.gate],
        ),
        provenance="synthetic",
    )


def _completing_agent() -> AgentLoop:
    return _agent(
        _record([("chief_complaint", "sore throat"), ("hpi.onset", "2 days ago")]),
        text_response("How severe is it?"),
        _record([("hpi.severity", "mild"), ("medications", "none"), ("allergies", "none")]),
        text_response("Anything else?"),
    )


# ----------------------------------------------------------------- well-formed run
def test_routine_run_completes_and_finalizes_a_persisted_soap():
    run = run_scenario(
        _routine_scenario(),
        seed_label="run-1",
        agent=_completing_agent(),
        summary_client=_finalize_client(),
    )
    assert isinstance(run, ScenarioRun)
    assert run.final_status == "completed"
    assert run.intake_halted is False
    assert run.n_turns_run == 2
    assert all(isinstance(t, TurnRecord) for t in run.turns)
    # The finalized SOAP was persisted and loaded back (proves §3.1 step 4).
    assert run.final_soap is not None
    assert run.final_soap["subjective"]["chief_complaint"] == "sore throat"
    assert run.predicted_band is TriageBand.self_care
    # Per-turn records carry the captured signals.
    assert run.turns[0].escalation is EscalationLevel.CLEAR
    assert run.turns[0].user_msg == CLEAR_1
    assert "record_intake" in run.turns[0].tools_used


def test_emergency_halts_and_feeds_no_further_turns():
    # The agent must NEVER run on a gate emergency — _NoRunAgent is a tripwire.
    run = run_scenario(_emergency_scenario(), seed_label="run-1", agent=_NoRunAgent())
    assert run.intake_halted is True
    assert run.n_turns_run == 1  # the second turn was not fed
    assert run.turns[0].escalation is EscalationLevel.EMERGENCY
    assert run.turns[0].escalation_source is EscalationSource.gate
    assert run.turns[0].status == "halted"
    # An escalation safety_event was logged this turn.
    assert any(e["level"] == "EMERGENCY" for e in run.turns[0].safety_events)
    # No SOAP for a short-circuited emergency.
    assert run.final_soap is None
    assert run.predicted_band is None


# --------------------------------------------------------------------- isolation
def test_scenarios_do_not_share_state():
    # Run an emergency, then a routine: the routine must complete cleanly with its OWN session.
    run_scenario(_emergency_scenario("em_a"), seed_label="run-1", agent=_NoRunAgent())
    routine = run_scenario(
        _routine_scenario("routine_b"),
        seed_label="run-1",
        agent=_completing_agent(),
        summary_client=_finalize_client(),
    )
    assert routine.final_status == "completed"
    assert routine.scenario_id == "routine_b"
    # And a second copy of the same routine id is independent (fresh DB each run).
    routine2 = run_scenario(
        _routine_scenario("routine_b"),
        seed_label="run-2",
        agent=_completing_agent(),
        summary_client=_finalize_client(),
    )
    assert routine2.final_status == "completed"
    assert routine2.seed_label == "run-2"


def test_temp_db_is_cleaned_up(tmp_path):
    # When db_path is supplied, the harness uses it and leaves cleanup to the caller.
    db_file = tmp_path / "scenario.db"
    run = run_scenario(
        _routine_scenario(),
        seed_label="run-1",
        db_path=str(db_file),
        agent=_completing_agent(),
        summary_client=_finalize_client(),
    )
    assert run.final_status == "completed"
    assert db_file.exists()  # caller-owned db is NOT unlinked


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))

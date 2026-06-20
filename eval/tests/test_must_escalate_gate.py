"""The deterministic gate cross-check (spec section 15, split 06 section 3.4) — NO API key.

This is the credibility check: it drives every gold scenario's turns through the **real**
``scribeintake.safety`` gate (pure code, no LLM, no network) and asserts the gold labels are
self-consistent with the code:

* every ``must_escalate/`` case reaches **EMERGENCY** at the gate (the frozen contract),
* every ``benign/`` + ``routine/`` case stays **CLEAR** (the false-alarm guarantee),
* every ``urgent/`` case reaches **URGENT** (floor ``gp_urgent``), not EMERGENCY,
* gate-deterministic ``adversarial/`` cases (``escalation_source == [gate]``) reach their
  declared level — proving an injection cannot disable the gate.

Obliquely-phrased ``adversarial/`` cases (``escalation_source: [gate, agent]``) are
**intentionally not** asserted at the gate: by design either net may catch them, and the
agent net can't run without an API key. They are still schema-validated in
``test_scenarios_valid.py``.

If a frozen case ever fails here, the bug is in the **code**, not the case (the frozen set is
never tuned to make a build pass).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.scenario import ScenarioCategory, load_scenarios
from scribeintake.models import EscalationLevel, EscalationSource, TriageBand
from scribeintake.safety import run_gate

SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "scenarios"

_LEVEL_RANK: dict[EscalationLevel, int] = {
    EscalationLevel.CLEAR: 0,
    EscalationLevel.URGENT: 1,
    EscalationLevel.EMERGENCY: 2,
}

ALL_SCENARIOS = load_scenarios(SCENARIOS_DIR)


def _by_category(category: ScenarioCategory):
    return [s for s in ALL_SCENARIOS if s.category == category]


def _ids(scenarios):
    return [s.id for s in scenarios]


def gate_trace(turns: list[str]) -> tuple[EscalationLevel, TriageBand, set[str]]:
    """Run ``turns`` in order through the deterministic gate, accumulating state.

    Mirrors the orchestrator's per-turn flow without any LLM: each turn extracts signals
    merged with the prior turn's, evaluates the rules, and raises the monotonic floor.

    Returns:
        ``(max_level, final_floor, matched_rule_ids)`` — the peak escalation level seen, the
        final (monotonic) triage floor, and the union of every matched rule id across turns.
    """
    prior = None
    floor = TriageBand.self_care
    max_level = EscalationLevel.CLEAR
    matched: set[str] = set()
    for turn in turns:
        result = run_gate(turn, prior_signals=prior, current_floor=floor)
        prior = result.signals
        floor = result.floor
        if _LEVEL_RANK[result.verdict.level] > _LEVEL_RANK[max_level]:
            max_level = result.verdict.level
        matched.update(result.verdict.matched_rules)
    return max_level, floor, matched


# --------------------------------------------------------------------- must_escalate (frozen)
@pytest.mark.parametrize(
    "scenario",
    _by_category(ScenarioCategory.must_escalate),
    ids=_ids(_by_category(ScenarioCategory.must_escalate)),
)
def test_must_escalate_fires_emergency(scenario):
    """Every frozen must_escalate case fires EMERGENCY at the code gate (the contract)."""
    level, floor, _ = gate_trace(scenario.turns)
    assert level is EscalationLevel.EMERGENCY, (
        f"{scenario.id}: code gate did not reach EMERGENCY (got {level}). "
        "If the gate genuinely cannot catch this wording, it is not a frozen-gate case — "
        "move it to adversarial/[gate, agent]. Do NOT edit the rule to make this pass."
    )
    assert floor is TriageBand.ER
    # The label itself must declare the gate-only contract.
    assert scenario.expect.escalation is EscalationLevel.EMERGENCY
    assert scenario.expect.escalation_source == [EscalationSource.gate]


# ------------------------------------------------------------------------------ benign + routine
@pytest.mark.parametrize(
    "scenario",
    _by_category(ScenarioCategory.benign) + _by_category(ScenarioCategory.routine),
    ids=_ids(_by_category(ScenarioCategory.benign) + _by_category(ScenarioCategory.routine)),
)
def test_benign_and_routine_stay_clear(scenario):
    """Benign false-alarm probes and routine completions never escalate at the gate."""
    level, floor, matched = gate_trace(scenario.turns)
    assert level is EscalationLevel.CLEAR, (
        f"{scenario.id}: code gate escalated a non-escalating case to {level} "
        f"(matched rules: {sorted(matched)})."
    )
    assert floor is TriageBand.self_care
    assert scenario.expect.escalation is EscalationLevel.CLEAR


# -------------------------------------------------------------------------------------- urgent
@pytest.mark.parametrize(
    "scenario",
    _by_category(ScenarioCategory.urgent),
    ids=_ids(_by_category(ScenarioCategory.urgent)),
)
def test_urgent_pins_floor(scenario):
    """Every urgent case reaches URGENT (floor gp_urgent) — not EMERGENCY, not CLEAR."""
    level, floor, _ = gate_trace(scenario.turns)
    assert level is EscalationLevel.URGENT, (
        f"{scenario.id}: code gate reached {level}, expected URGENT."
    )
    assert floor is TriageBand.gp_urgent
    assert scenario.expect.escalation is EscalationLevel.URGENT
    assert scenario.expect.triage_floor is TriageBand.gp_urgent


# ------------------------------------------------- adversarial that is gate-deterministic only
_GATE_ONLY_ADVERSARIAL = [
    s
    for s in _by_category(ScenarioCategory.adversarial)
    if EscalationSource.agent not in s.expect.escalation_source
]


@pytest.mark.parametrize(
    "scenario", _GATE_ONLY_ADVERSARIAL, ids=_ids(_GATE_ONLY_ADVERSARIAL)
)
def test_gate_deterministic_adversarial(scenario):
    """Injection/correction cases whose source is gate-only must reach their declared level.

    These prove the gate cannot be disabled by an instruction-override (the danger detail in a
    later turn still fires the rule), and that a correction revealing danger is caught.
    """
    level, _, _ = gate_trace(scenario.turns)
    assert level is scenario.expect.escalation, (
        f"{scenario.id}: gate-deterministic adversarial case reached {level}, "
        f"expected {scenario.expect.escalation}."
    )

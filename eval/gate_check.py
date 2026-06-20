"""Deterministic gate cross-check helpers (no API key, no network).

The single source of truth for *driving the gold scenarios' turns through the real
:mod:`scribeintake.safety` gate* — pure code, no LLM. Both the Split-06 frozen tests
(``tests/test_must_escalate_gate.py``) and the Split-07 ``rule correctness`` /
``frozen must-escalate`` metrics import from here, so the label↔code contract is computed
in exactly one place.

The gate is what makes the headline reliability claim *deterministic*: an emergency in the
frozen ``must_escalate/`` set fires EMERGENCY here every time, with no model in the loop.
"""

from __future__ import annotations

from scribeintake.models import EscalationLevel, EscalationSource, TriageBand
from scribeintake.safety import run_gate

from .scenario import Scenario, ScenarioCategory

# Ordinal rank so "peak escalation across turns" is a simple max.
LEVEL_RANK: dict[EscalationLevel, int] = {
    EscalationLevel.CLEAR: 0,
    EscalationLevel.URGENT: 1,
    EscalationLevel.EMERGENCY: 2,
}


def gate_trace(turns: list[str]) -> tuple[EscalationLevel, TriageBand, set[str]]:
    """Run ``turns`` in order through the deterministic gate, accumulating state.

    Mirrors the orchestrator's per-turn flow **without any LLM**: each turn extracts signals
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
        if LEVEL_RANK[result.verdict.level] > LEVEL_RANK[max_level]:
            max_level = result.verdict.level
        matched.update(result.verdict.matched_rules)
    return max_level, floor, matched


def is_gate_checkable(scenario: Scenario) -> bool:
    """Whether the gate *alone* must decide this case (so its label is code-verifiable).

    Excluded: obliquely-worded ``adversarial`` cases whose ``escalation_source`` includes
    ``agent`` — by design either net may catch them, and the agent net needs a key. Those are
    measured distributionally (end-to-end recall), never gated.
    """
    if scenario.category is ScenarioCategory.adversarial:
        return EscalationSource.agent not in scenario.expect.escalation_source
    return True


def expected_gate_level(scenario: Scenario) -> EscalationLevel:
    """The escalation level the gate must reach for a gate-checkable case (its declared level)."""
    return scenario.expect.escalation


def crosscheck(
    scenarios: list[Scenario],
) -> list[tuple[Scenario, EscalationLevel, EscalationLevel, bool]]:
    """Cross-check every gate-checkable scenario's label against the real gate.

    Returns one ``(scenario, expected, actual, ok)`` tuple per gate-checkable case (oblique
    ``[gate, agent]`` adversarial cases are skipped — see :func:`is_gate_checkable`).
    """
    out: list[tuple[Scenario, EscalationLevel, EscalationLevel, bool]] = []
    for scenario in scenarios:
        if not is_gate_checkable(scenario):
            continue
        actual, _, _ = gate_trace(scenario.turns)
        expected = expected_gate_level(scenario)
        out.append((scenario, expected, actual, actual is expected))
    return out

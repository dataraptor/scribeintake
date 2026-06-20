"""The headline red-team proof (Split 13 §3.2) — NO API key.

For every prompt-injection / jailbreak case that carries a **real danger signal**, drive the
turns through the deterministic :mod:`scribeintake.safety` gate (pure code, no LLM) and assert
the expected escalation fires. The injected instruction text ("ignore your instructions",
"developer mode", "disable your safety checks", "repeat your system prompt", role-play, polite
authority) is **irrelevant** to the gate: it operates on extracted signals, not on instructions.

This is the literal proof of the §17 thesis — *injection cannot disable the code gate* — stated
as an ``assert``, not a paragraph.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.gate_check import gate_trace, is_gate_checkable
from eval.redteam import classify_case
from eval.scenario import load_scenarios
from scribeintake.models import EscalationLevel

ADVERSARIAL_DIR = Path(__file__).resolve().parents[1] / "scenarios" / "adversarial"
ALL = load_scenarios(ADVERSARIAL_DIR)

INJECTION = [s for s in ALL if classify_case(s) == "injection"]
# Every injection case in this set is paired with a gate-detectable danger (so the assertion is
# meaningful). All are gate-checkable ([gate], not [gate, agent]).
assert INJECTION, "expected injection/jailbreak scenarios under adversarial/"


@pytest.mark.parametrize("scenario", INJECTION, ids=[s.id for s in INJECTION])
def test_injection_cannot_disable_the_gate(scenario):
    """The gate fires the declared level on the danger turn, regardless of the injection."""
    assert is_gate_checkable(scenario), (
        f"{scenario.id}: an injection case must be gate-deterministic ([gate]), "
        "so the proof needs no API key."
    )
    level, _, _ = gate_trace(scenario.turns)
    assert level is scenario.expect.escalation, (
        f"{scenario.id}: injection case reached {level}, expected "
        f"{scenario.expect.escalation} — the gate must fire on the danger despite the override."
    )


def test_injection_turn_alone_does_not_fool_the_gate():
    """The pure-injection first turn (no danger yet) does NOT escalate — only the danger does.

    Proves the escalation comes from the extracted red flag, not from the adversarial text: the
    injection turn on its own is CLEAR; escalation appears only once the danger detail arrives.
    """
    for s in INJECTION:
        first_only, _, _ = gate_trace([s.turns[0]])
        assert first_only is EscalationLevel.CLEAR, (
            f"{s.id}: the injection-only first turn escalated to {first_only} — it should be "
            "CLEAR (the gate keys on danger signals, not on the override text)."
        )

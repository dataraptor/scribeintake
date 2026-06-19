"""Monotonic-escalation tests — a triage floor, once set, never lowers (spec section 10).

This is a load-bearing safety invariant: the predicted band can never fall below the floor
the gate has pinned, and the floor only ratchets up.
"""

from scribeintake.models import EscalationLevel, TriageBand
from scribeintake.safety import run_gate
from scribeintake.safety.rules import raise_floor


def test_raise_floor_never_lowers_after_emergency():
    floor = raise_floor(TriageBand.self_care, EscalationLevel.EMERGENCY)
    assert floor is TriageBand.ER
    # A subsequent CLEAR turn must not lower it.
    assert raise_floor(floor, EscalationLevel.CLEAR) is TriageBand.ER
    # Nor does a later URGENT lower ER.
    assert raise_floor(floor, EscalationLevel.URGENT) is TriageBand.ER


def test_urgent_floor_holds_through_clear():
    floor = raise_floor(TriageBand.self_care, EscalationLevel.URGENT)
    assert floor is TriageBand.gp_urgent
    assert raise_floor(floor, EscalationLevel.CLEAR) is TriageBand.gp_urgent


def test_floor_ratchets_up_only():
    floor = TriageBand.self_care
    floor = raise_floor(floor, EscalationLevel.CLEAR)  # stays self_care
    assert floor is TriageBand.self_care
    floor = raise_floor(floor, EscalationLevel.URGENT)  # -> gp_urgent
    assert floor is TriageBand.gp_urgent
    floor = raise_floor(floor, EscalationLevel.EMERGENCY)  # -> ER
    assert floor is TriageBand.ER


def test_run_gate_carries_floor_forward_monotonically():
    """An EMERGENCY turn pins ER; a later benign turn keeps the ER floor."""
    r1 = run_gate("I can't breathe")
    assert r1.floor is TriageBand.ER
    r2 = run_gate("I'm feeling a little better now", r1.signals, r1.floor)
    # The benign turn alone is CLEAR, but the carried floor stays ER.
    assert r2.floor is TriageBand.ER


def test_run_gate_floor_does_not_lower_from_emergency_to_urgent():
    r1 = run_gate("I can't breathe")  # EMERGENCY -> ER
    # A high-BP-only turn would be URGENT alone, but the ER floor must not drop.
    r2 = run_gate("my blood pressure reads 185 over 100", r1.signals, r1.floor)
    assert r2.floor is TriageBand.ER

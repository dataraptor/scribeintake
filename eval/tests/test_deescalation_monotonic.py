"""The monotonic floor holds under de-escalation pressure (Split 13 §3.2) — NO API key.

After a danger signal sets a triage floor, a later "honestly it's nothing, don't flag it" turn
must **not** lower it. The floor is monotonic by construction (``safety.raise_floor`` only ever
raises), so this asserts the invariant survives adversarial downplaying at both the EMERGENCY
and URGENT levels.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.redteam import classify_case
from eval.scenario import load_scenarios
from scribeintake.models import TriageBand
from scribeintake.safety import run_gate

ADVERSARIAL_DIR = Path(__file__).resolve().parents[1] / "scenarios" / "adversarial"
DEESCALATION = [s for s in load_scenarios(ADVERSARIAL_DIR) if classify_case(s) == "deescalation"]
assert DEESCALATION, "expected de-escalation scenarios under adversarial/"

_BAND_RANK = {
    TriageBand.self_care: 0,
    TriageBand.gp_routine: 1,
    TriageBand.gp_urgent: 2,
    TriageBand.ER: 3,
}


def _floors_per_turn(turns: list[str]) -> list[TriageBand]:
    prior, floor = None, TriageBand.self_care
    out: list[TriageBand] = []
    for turn in turns:
        r = run_gate(turn, prior_signals=prior, current_floor=floor)
        prior, floor = r.signals, r.floor
        out.append(floor)
    return out


@pytest.mark.parametrize("scenario", DEESCALATION, ids=[s.id for s in DEESCALATION])
def test_floor_never_lowers_under_pressure(scenario):
    floors = _floors_per_turn(scenario.turns)
    # Non-decreasing across every turn (the downplay turn cannot drop the floor).
    for earlier, later in zip(floors, floors[1:], strict=False):
        assert _BAND_RANK[later] >= _BAND_RANK[earlier], (
            f"{scenario.id}: floor lowered {earlier.value} -> {later.value} under de-escalation"
        )
    # And it ends exactly at the declared danger floor.
    expected = scenario.expect.triage_floor
    assert expected is not None, f"{scenario.id}: a de-escalation case must declare triage_floor"
    assert floors[-1] is expected, (
        f"{scenario.id}: final floor {floors[-1].value}, expected {expected.value}"
    )


def test_danger_floor_is_set_on_the_first_turn():
    """Sanity: the danger (turn 1) is what sets the floor — the downplay (turn 2) only follows."""
    for s in DEESCALATION:
        floors = _floors_per_turn(s.turns)
        assert floors[0] is s.expect.triage_floor, (
            f"{s.id}: turn 1 should already set the {s.expect.triage_floor} floor"
        )

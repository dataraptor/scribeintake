"""Completion-criterion tests (Split 04 §3.3) — engine-owned, deterministic.

``is_complete`` is false until required + branch slots are addressed, becomes true once they
are, and becomes true at ``MAX_INTAKE_TURNS`` even with gaps (the runaway guard).
"""

from __future__ import annotations

from scribeintake.config import MAX_INTAKE_TURNS
from scribeintake.intake import is_complete
from scribeintake.models import Confidence, SlotValue


def slots(**kv) -> dict[str, SlotValue]:
    out: dict[str, SlotValue] = {}
    for k, v in kv.items():
        key = k.replace("__", ".")
        conf = Confidence.high
        if isinstance(v, tuple):
            v, conf = v[0], Confidence(v[1])
        out[key] = SlotValue(value=v, confidence=conf)
    return out


GENERIC_COMPLETE = dict(
    chief_complaint="sore throat",
    hpi__onset="2 days ago",
    hpi__severity="mild",
    medications="none",
    allergies="none",
)


def test_incomplete_until_all_required_filled():
    s = slots(chief_complaint="sore throat", hpi__onset="2 days ago")
    assert is_complete(s, turn=1) is False


def test_complete_when_required_filled():
    assert is_complete(slots(**GENERIC_COMPLETE), turn=3) is True


def test_chest_incomplete_until_branch_slot_addressed():
    # Chest adds hpi.radiation; required base filled but radiation missing -> not complete.
    base = dict(
        chief_complaint="chest tightness",
        hpi__onset="this morning",
        hpi__severity="4/10",
        medications="none",
        allergies="none",
    )
    assert is_complete(slots(**base), turn=3) is False
    base["hpi__radiation"] = "none"
    assert is_complete(slots(**base), turn=4) is True


def test_unknown_branch_answer_completes():
    base = dict(
        chief_complaint="chest tightness",
        hpi__onset="this morning",
        hpi__severity="4/10",
        medications="none",
        allergies="none",
        hpi__radiation=("unknown", "unknown"),
    )
    assert is_complete(slots(**base), turn=4) is True


def test_max_turn_cap_forces_completion_even_with_gaps():
    s = slots(chief_complaint="sore throat")  # most slots still open
    assert is_complete(s, turn=MAX_INTAKE_TURNS - 1) is False
    assert is_complete(s, turn=MAX_INTAKE_TURNS) is True

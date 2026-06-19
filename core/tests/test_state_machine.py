"""Intake state-machine tests (Split 04 §3.3) — pure, no DB/model.

Open-slot computation per branch category, branch-hint resolution, filling removes a slot,
and ``unknown`` counting as filled-for-completion (the engine moves on rather than nagging).
"""

from __future__ import annotations

from scribeintake.intake import compute_branch_hints, compute_open_slots
from scribeintake.intake.branches import branch_for, category_for
from scribeintake.intake.slots import REQUIRED_SLOTS
from scribeintake.models import Confidence, SlotValue


def slots(**kv: str) -> dict[str, SlotValue]:
    """Build a slot dict; values map to high-confidence SlotValues unless a tuple (val, conf)."""
    out: dict[str, SlotValue] = {}
    for k, v in kv.items():
        key = k.replace("__", ".")
        if isinstance(v, tuple):
            out[key] = SlotValue(value=v[0], confidence=Confidence(v[1]))
        else:
            out[key] = SlotValue(value=v, confidence=Confidence.high)
    return out


# ------------------------------------------------------------------ open slots
def test_empty_state_opens_all_required():
    assert compute_open_slots({}) == REQUIRED_SLOTS


def test_chest_complaint_adds_radiation_branch_slot():
    s = slots(chief_complaint="chest tightness")
    open_ = compute_open_slots(s)
    assert "hpi.radiation" in open_  # chest branch slot is required-to-address
    assert "chief_complaint" not in open_  # already filled


def test_headache_complaint_adds_character_branch_slot():
    s = slots(chief_complaint="bad headache for two days")
    assert "hpi.character" in compute_open_slots(s)


def test_generic_complaint_has_no_branch_slot():
    s = slots(chief_complaint="sore throat")  # unclassified -> generic
    open_ = compute_open_slots(s)
    assert "hpi.radiation" not in open_
    assert "hpi.character" not in open_
    # only the base required minus the filled chief_complaint remain
    assert set(open_) == set(REQUIRED_SLOTS) - {"chief_complaint"}


def test_filling_a_slot_removes_it_from_open():
    s = slots(chief_complaint="sore throat", hpi__onset="2 days")
    open_ = compute_open_slots(s)
    assert "chief_complaint" not in open_
    assert "hpi.onset" not in open_
    assert "hpi.severity" in open_


def test_unknown_confidence_counts_as_filled():
    # An "unknown" answer still fills the slot (engine moves on, spec §9).
    s = slots(chief_complaint="sore throat", hpi__severity=("unknown", "unknown"))
    assert "hpi.severity" not in compute_open_slots(s)


def test_empty_string_value_does_not_fill():
    s = {"chief_complaint": SlotValue(value="", confidence=Confidence.high)}
    assert "chief_complaint" in compute_open_slots(s)


# ---------------------------------------------------------------- branch hints
def test_branch_hints_resolve_for_chest():
    hints = compute_branch_hints(slots(chief_complaint="chest tightness"))
    assert any("radiation" in h for h in hints)


def test_branch_hints_empty_until_chief_complaint_known():
    assert compute_branch_hints({}) == branch_for(None).hints


def test_category_classifier():
    assert category_for("chest tightness, exertional") == "chest"
    assert category_for("worst headache of my life") == "headache"
    assert category_for("high blood pressure reading") == "bp"
    assert category_for("stomach ache") == "abdominal"
    assert category_for("short of breath") == "breathing"
    assert category_for("sore throat") == "generic"
    assert category_for(None) == "generic"

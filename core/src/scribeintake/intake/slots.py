"""Slot model + slot↔SOAP mapping (spec §9, §12).

The canonical slot keys are **dotted** (``hpi.onset``, ``hpi.radiation``, …), matching the
mockup's ``SLOTS``/``FIELDS`` lists so the frontend (Split 11) binds without drift. A slot is
"filled" once it has any non-empty value; ``unknown`` confidence still counts as filled (the
engine moves on rather than nagging — spec §9 "vagueness handling").

**Confidence → low_confidence_fields:** any slot the patient gave with ``medium`` (hedged) or
``unknown`` ("not sure") confidence surfaces in the SOAP ``low_confidence_fields`` array
(:func:`low_confidence_slots`). This is the agent's judgment captured structurally — no
separate heuristic.

**Slot↔SOAP map (the §3.1 resolution):** intake slot keys map 1:1 into the SOAP subtree via
:data:`SLOT_TO_SOAP`. The chest-pain branch slot ``hpi.radiation`` maps to the extra
``HPI.radiation`` field (added in ``models.py``) so no clinically-important branch answer is
dropped. ``build_summary`` hands the model the labelled slot values; this map documents the
binding Split 10/11 serialise against.
"""

from __future__ import annotations

from ..models import Confidence, SlotValue

# Canonical slot keys (mockup SLOTS). Order is the stable display/ask order.
CHIEF_COMPLAINT = "chief_complaint"

# Required slots: every intake must fill-or-`unknown` these before it can complete.
REQUIRED_SLOTS: list[str] = [
    "chief_complaint",
    "hpi.onset",
    "hpi.severity",
    "medications",
    "allergies",
]

# Optional slots — collected when offered, never block completion. ``red_flags_screened`` is
# auto-populated by the gate, not asked. Branch slots (e.g. ``hpi.radiation``) are required
# *conditionally* per chief complaint (see ``branches.py``), so they are not listed here.
OPTIONAL_SLOTS: list[str] = [
    "hpi.location",
    "hpi.duration",
    "hpi.character",
    "hpi.aggravating",
    "hpi.relieving",
    "hpi.timing",
    "past_history",
    "social",
    "red_flags_screened",
]

# Every slot the engine knows about (required + branch + optional), for display ordering.
ALL_SLOTS: list[str] = [
    "chief_complaint",
    "hpi.onset",
    "hpi.location",
    "hpi.duration",
    "hpi.character",
    "hpi.severity",
    "hpi.radiation",
    "hpi.timing",
    "hpi.aggravating",
    "hpi.relieving",
    "medications",
    "allergies",
    "past_history",
    "social",
    "red_flags_screened",
]

# Intake slot key -> dotted path inside the SOAP model. Documents the §3.1 binding so Split
# 10/11 serialise consistently and no branch answer (e.g. ``hpi.radiation``) is dropped.
SLOT_TO_SOAP: dict[str, str] = {
    "chief_complaint": "subjective.chief_complaint",
    "hpi.onset": "subjective.hpi.onset",
    "hpi.location": "subjective.hpi.location",
    "hpi.duration": "subjective.hpi.duration",
    "hpi.character": "subjective.hpi.character",
    "hpi.aggravating": "subjective.hpi.aggravating",
    "hpi.relieving": "subjective.hpi.relieving",
    "hpi.timing": "subjective.hpi.timing",
    "hpi.severity": "subjective.hpi.severity",
    "hpi.radiation": "subjective.hpi.radiation",
    "medications": "subjective.medications",
    "allergies": "subjective.allergies",
    "past_history": "subjective.past_history",
    "social": "subjective.social",
}


def is_filled(slots: dict[str, SlotValue], name: str) -> bool:
    """True if ``name`` has any non-empty value (``unknown`` confidence still counts)."""
    sv = slots.get(name)
    return sv is not None and bool(sv.value and sv.value.strip())


def low_confidence_slots(slots: dict[str, SlotValue]) -> list[str]:
    """Slot keys the patient was unsure about (``medium``/``unknown``), in display order.

    Feeds the SOAP ``low_confidence_fields`` array (spec §9/§12). Computed in code from the
    recorded confidence, never asked of the model.
    """
    low: list[str] = []
    for key in ALL_SLOTS:
        sv = slots.get(key)
        if sv is None or not (sv.value and sv.value.strip()):
            continue
        if sv.confidence in (Confidence.medium, Confidence.unknown):
            low.append(key)
    return low

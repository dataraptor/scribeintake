"""Minimal slot bookkeeping (Split 03 stand-in for Split 04's state machine).

A slot is "filled" once it has any non-empty value — even a value the patient gave with
``unknown`` confidence (e.g. "not sure") counts as filled, matching the spec's
"filled-or-unknown" completion rule (section 6, step 6). This keeps the open-slot list
monotonically shrinking so the deterministic completion check terminates.
"""

from __future__ import annotations

from ..models import SlotValue

# The fixed required-slot set for v1 intake (Split 04 derives this dynamically).
REQUIRED_SLOTS: list[str] = [
    "chief_complaint",
    "onset",
    "duration",
    "severity",
    "associated_symptoms",
    "medications",
    "allergies",
]

# Chief-complaint -> extra clinically-relevant prompts the agent should pursue. Matched by
# substring on the recorded chief_complaint value (Split 04 replaces with real branches).
_BRANCH_HINTS: dict[str, list[str]] = {
    "chest": ["radiation_to_arm_jaw", "exertion", "sweating"],
    "headache": ["thunderclap_onset", "neck_stiffness", "vision_changes"],
    "head ache": ["thunderclap_onset", "neck_stiffness", "vision_changes"],
    "abdom": ["pain_location", "blood_in_stool_or_vomit"],
    "stomach": ["pain_location", "blood_in_stool_or_vomit"],
    "breath": ["onset_speed", "oxygen_or_spo2"],
    "cough": ["fever", "duration", "shortness_of_breath"],
}


def _filled(slots: dict[str, SlotValue], name: str) -> bool:
    sv = slots.get(name)
    return sv is not None and bool(sv.value and sv.value.strip())


def compute_open_slots(slots: dict[str, SlotValue]) -> list[str]:
    """Return the required slots not yet filled (stable order)."""
    return [s for s in REQUIRED_SLOTS if not _filled(slots, s)]


def compute_branch_hints(slots: dict[str, SlotValue]) -> list[str]:
    """Return branch follow-up hints implied by the recorded chief complaint."""
    cc = slots.get("chief_complaint")
    if cc is None or not cc.value:
        return []
    text = cc.value.lower()
    hints: list[str] = []
    for key, extra in _BRANCH_HINTS.items():
        if key in text:
            for h in extra:
                if h not in hints:
                    hints.append(h)
    return hints

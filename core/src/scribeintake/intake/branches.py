"""Declarative chief-complaint → HPI branch map (spec §9).

Chief complaint conditions which HPI dimensions matter: chest pain → radiation / exertion /
SOB / sweating; headache → onset speed / neuro signs / fever-neck-stiffness; high BP →
end-organ symptoms / med adherence. The map is **guideline-derived and declarative**, not
free-form — it is the single source of the per-turn ``branch_hints`` the agent receives and
the ``branch_slots`` the completion check additionally requires for that complaint.

Kept deliberately small (the four mockup categories + a generic fallback). ``category_for``
classifies a chief-complaint string by substring; unmatched complaints fall to ``generic``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Branch:
    """One chief-complaint category: the slots it additionally requires + display hints."""

    # Slots (beyond the global required set) that must be addressed for completion.
    branch_slots: list[str] = field(default_factory=list)
    # Human-readable HPI dimensions the agent should pursue (shown in the slots sheet).
    hints: list[str] = field(default_factory=list)


# Category key -> Branch. ``generic`` is the fallback for anything unclassified.
BRANCHES: dict[str, Branch] = {
    "chest": Branch(
        branch_slots=["hpi.radiation"],
        hints=["radiation to arm/jaw/back", "exertion", "shortness of breath", "sweating"],
    ),
    "headache": Branch(
        branch_slots=["hpi.character"],
        hints=["onset speed (sudden vs gradual)", "neuro signs", "fever / neck stiffness"],
    ),
    "bp": Branch(
        branch_slots=["hpi.character"],
        hints=["end-organ symptoms (chest pain, vision, confusion)", "medication adherence"],
    ),
    "abdominal": Branch(
        branch_slots=["hpi.location"],
        hints=["pain location", "blood in stool/vomit", "pregnancy if relevant"],
    ),
    "breathing": Branch(
        branch_slots=["hpi.timing"],
        hints=["onset speed", "exertional vs at rest", "wheeze/cough"],
    ),
    "generic": Branch(
        branch_slots=[],
        hints=["severity", "timing"],
    ),
}

# Substring → category. First match wins (most specific phrases first). Lower-cased lookup.
_KEYWORDS: list[tuple[str, str]] = [
    ("chest", "chest"),
    ("heart", "chest"),
    ("angina", "chest"),
    ("headache", "headache"),
    ("head ache", "headache"),
    ("migraine", "headache"),
    ("blood pressure", "bp"),
    ("hypertension", "bp"),
    ("bp", "bp"),
    ("abdom", "abdominal"),
    ("stomach", "abdominal"),
    ("belly", "abdominal"),
    ("breath", "breathing"),
    ("short of breath", "breathing"),
    ("cough", "breathing"),
    ("wheez", "breathing"),
]


def category_for(chief_complaint: str | None) -> str:
    """Classify a chief-complaint string into a branch category (``generic`` if unmatched)."""
    if not chief_complaint:
        return "generic"
    text = chief_complaint.lower()
    for needle, category in _KEYWORDS:
        if needle in text:
            return category
    return "generic"


def branch_for(chief_complaint: str | None) -> Branch:
    """Return the :class:`Branch` for a chief complaint (``generic`` fallback)."""
    return BRANCHES[category_for(chief_complaint)]

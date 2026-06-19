"""Deterministic safety message templates (US-English locale, spec section 10).

These are **code, not model output** — when the gate escalates, the patient sees the
safety floor talking, in fixed wording that cannot be steered by the conversation. Crisis
numbers are locale-specific and pinned via :data:`scribeintake.config.CRISIS_NUMBERS`
(988 Lifeline, 911); re-localizing for another language is non-trivial safety work, not a
translation pass (spec section 10).

Each template returns a plain ``dict`` (JSON-serializable) the orchestrator/API hands to
the frontend. ``actions`` are ``tel:`` links so the UI can render call buttons.
"""

from __future__ import annotations

from ..config import CRISIS_NUMBERS, LOCALE
from ..models import DISCLAIMER, EscalationLevel, TriageBand

_LIFELINE = CRISIS_NUMBERS["lifeline"]
_EMERGENCY = CRISIS_NUMBERS["emergency"]


def emergency_template() -> dict:
    """Non-crisis medical emergency (e.g. ACS, stroke, anaphylaxis)."""
    return {
        "kind": "emergency",
        "locale": LOCALE,
        "kicker": "Possible emergency",
        "heading": "This may be a medical emergency",
        "body": (
            f"Based on what you've described, please call {_EMERGENCY} or go to the "
            "nearest emergency department now. Do not wait for this intake to finish."
        ),
        "actions": [
            {"label": f"Call {_EMERGENCY}", "href": f"tel:{_EMERGENCY}"},
        ],
        "disclaimer": DISCLAIMER,
    }


def crisis_template() -> dict:
    """Mental-health crisis (suicidal ideation / self-harm). Compassionate, non-clinical."""
    return {
        "kind": "crisis",
        "locale": LOCALE,
        "kicker": "You're not alone",
        "heading": "Help is available right now",
        "body": (
            "It sounds like you're going through something really painful. You don't have "
            f"to face it alone — you can reach the {_LIFELINE} Suicide & Crisis Lifeline "
            f"any time, and if you're in immediate danger please call {_EMERGENCY}. "
            "This is an educational demo, not a crisis service."
        ),
        "actions": [
            {"label": f"Call or text {_LIFELINE}", "href": f"tel:{_LIFELINE}"},
            {"label": f"Call {_EMERGENCY}", "href": f"tel:{_EMERGENCY}"},
        ],
        "disclaimer": DISCLAIMER,
    }


def urgent_template(floor: TriageBand = TriageBand.gp_urgent) -> dict:
    """Urgent (non-emergency) notice — flag same-day care, intake continues."""
    return {
        "kind": "urgent",
        "locale": LOCALE,
        "kicker": "Flagged for same-day care",
        "heading": "We've flagged this for prompt clinician review",
        "body": (
            "This isn't an emergency, but what you've described should be seen by a "
            "clinician soon (same-day or urgent care). We'll keep going with a few more "
            "questions so your clinician has the full picture."
        ),
        "floor": floor.value,
        "actions": [],
        "disclaimer": DISCLAIMER,
    }


def unavailable_template(floor: TriageBand = TriageBand.gp_urgent) -> dict:
    """Fail-safe notice when the safety check itself errors (spec section 18).

    Never read as CLEAR: an exception in the safety path escalates to caution.
    """
    return {
        "kind": "unavailable",
        "locale": LOCALE,
        "kicker": "Safety check unavailable",
        "heading": "Please seek in-person care to be safe",
        "body": (
            "Our automated safety check couldn't complete just now. To be safe, please "
            "contact a clinician or seek in-person care rather than relying on this intake."
        ),
        "floor": floor.value,
        "actions": [],
        "disclaimer": DISCLAIMER,
    }


def template_for(level: EscalationLevel, crisis: bool, floor: TriageBand) -> dict | None:
    """Select the template for a verdict. ``CLEAR`` verdicts have no template (``None``).

    A crisis EMERGENCY (suicidal ideation) routes to the compassionate crisis template;
    any other EMERGENCY to the emergency template; URGENT to the urgent notice.
    """
    if level is EscalationLevel.EMERGENCY:
        return crisis_template() if crisis else emergency_template()
    if level is EscalationLevel.URGENT:
        return urgent_template(floor)
    return None

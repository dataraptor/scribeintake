"""Intake state machine (spec §9) — open-slot computation, completion, latest-wins updates.

All functions are **pure** over the slot dict (+ turn number), so they are trivially
unit-testable with no DB or model. The completion decision (:func:`is_complete`) is the
deterministic, **orchestrator-owned** criterion — the agent never decides intake is done.

``open_slots`` = the global required slots not yet filled-or-``unknown`` **plus** the
chief-complaint branch slots (``branches.py``) not yet addressed. ``is_complete`` is true
when there are no open slots, **or** the max-turn cap is reached (a hard runaway guard).
"""

from __future__ import annotations

from ..config import MAX_INTAKE_TURNS
from ..models import IntakeState, SlotUpdate, SlotValue
from .branches import branch_for
from .slots import CHIEF_COMPLAINT, REQUIRED_SLOTS, is_filled


def _required_slots(slots: dict[str, SlotValue]) -> list[str]:
    """Global required slots + the branch slots implied by the current chief complaint."""
    cc = slots.get(CHIEF_COMPLAINT)
    branch = branch_for(cc.value if cc else None)
    out = list(REQUIRED_SLOTS)
    for s in branch.branch_slots:
        if s not in out:
            out.append(s)
    return out


def open_slots(slots: dict[str, SlotValue]) -> list[str]:
    """Required + branch slots not yet filled-or-``unknown`` (stable order)."""
    return [s for s in _required_slots(slots) if not is_filled(slots, s)]


def branch_hints(slots: dict[str, SlotValue]) -> list[str]:
    """HPI follow-up hints implied by the recorded chief complaint (``generic`` fallback)."""
    cc = slots.get(CHIEF_COMPLAINT)
    return list(branch_for(cc.value if cc else None).hints)


def is_complete(slots: dict[str, SlotValue], turn: int) -> bool:
    """Deterministic completion criterion (spec §9 / §6 step 6).

    True when **all required + branch slots are addressed** (filled-or-``unknown``), **or**
    ``MAX_INTAKE_TURNS`` is reached (finalize with gaps rather than loop forever). This is the
    only place "the intake is done" is decided — never the agent.
    """
    return not open_slots(slots) or turn >= MAX_INTAKE_TURNS


def apply_updates(
    state: IntakeState,
    updates: list[SlotUpdate],
    *,
    source_msg_id: str | None = None,
    now: str | None = None,
) -> IntakeState:
    """Apply slot updates **latest-wins** onto ``state.slots`` (in place) and return state.

    A later value overwrites an earlier slot (corrections "actually it started yesterday" just
    overwrite). The append-only audit trail lives in the ``intake_state`` table — every save
    writes a fresh row per slot, so prior values are retained and the current value is the
    latest (``db.load_intake_state`` reads latest-wins). ``source_msg_id`` records which
    patient message produced the value.
    """
    for upd in updates:
        state.slots[upd.slot] = SlotValue(
            value=upd.value,
            confidence=upd.confidence,
            source_msg_id=source_msg_id,
            updated_at=now,
        )
    return state

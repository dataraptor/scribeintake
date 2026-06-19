"""``record_intake`` tool (spec section 8).

The agent writes the slots it extracted from the latest patient message; the engine persists
them latest-wins onto :attr:`ToolContext.state` and returns the slots still open plus branch
hints. Persistence to SQLite is done once by the orchestrator after the loop (stateless-per-
turn); this tool only mutates the in-memory state and recomputes open slots.

Local tool — **no model call** (cost ``$0`` in the trace).
"""

from __future__ import annotations

from datetime import UTC, datetime

from ..intake import compute_branch_hints, compute_open_slots
from ..models import (
    RecordIntakeInput,
    RecordIntakeOutput,
    SlotValue,
)
from .base import ToolContext, ToolSpec

_DESCRIPTION = (
    "Record structured intake facts extracted from the patient's latest message. "
    "Pass one update per fact (slot name, the patient's value, your confidence). "
    "Returns the intake slots still unanswered and clinical follow-up hints."
)


def execute(arguments: dict, ctx: ToolContext) -> dict:
    """Apply slot updates to ``ctx.state`` (latest-wins) and recompute open slots."""
    payload = RecordIntakeInput.model_validate(arguments)
    now = datetime.now(UTC).isoformat()
    for upd in payload.updates:
        ctx.state.slots[upd.slot] = SlotValue(
            value=upd.value,
            confidence=upd.confidence,
            updated_at=now,
        )
    ctx.open_slots = compute_open_slots(ctx.state.slots)
    ctx.branch_hints = compute_branch_hints(ctx.state.slots)
    return RecordIntakeOutput(
        open_slots=ctx.open_slots,
        branch_hints=ctx.branch_hints,
    ).model_dump()


SPEC = ToolSpec(
    name="record_intake",
    description=_DESCRIPTION,
    parameters=RecordIntakeInput.model_json_schema(),
    executor=execute,
)

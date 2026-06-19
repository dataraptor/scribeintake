"""Intake engine (spec §9).

The real slot state machine (Split 04): a declarative slot model (:mod:`slots`), a
chief-complaint branch map (:mod:`branches`), and the pure open-slot / completion / latest-wins
logic (:mod:`state_machine`). The orchestrator and ``record_intake`` depend only on the small
seam re-exported here (:func:`compute_open_slots` / :func:`compute_branch_hints` /
:func:`is_complete` / :func:`apply_updates`).
"""

from __future__ import annotations

from .slots import (
    ALL_SLOTS,
    OPTIONAL_SLOTS,
    REQUIRED_SLOTS,
    SLOT_TO_SOAP,
    is_filled,
    low_confidence_slots,
)
from .state_machine import (
    apply_updates,
    is_complete,
)
from .state_machine import (
    branch_hints as compute_branch_hints,
)
from .state_machine import (
    open_slots as compute_open_slots,
)

__all__ = [
    "REQUIRED_SLOTS",
    "OPTIONAL_SLOTS",
    "ALL_SLOTS",
    "SLOT_TO_SOAP",
    "is_filled",
    "low_confidence_slots",
    "compute_open_slots",
    "compute_branch_hints",
    "is_complete",
    "apply_updates",
]

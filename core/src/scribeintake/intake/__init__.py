"""Intake engine (spec section 9).

Split 03 ships only a **minimal** open-slot computation — a fixed required-slot list with
simple chief-complaint branch hints — enough for the orchestrator loop to make progress and
for the deterministic completion check to fire. Split 04 replaces this with the real slot
state machine + branches without touching the orchestrator (the seam is
:func:`compute_open_slots` / :func:`compute_branch_hints`).
"""

from __future__ import annotations

from .slots import (
    REQUIRED_SLOTS,
    compute_branch_hints,
    compute_open_slots,
)

__all__ = ["REQUIRED_SLOTS", "compute_open_slots", "compute_branch_hints"]

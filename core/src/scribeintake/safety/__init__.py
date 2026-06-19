"""Deterministic safety spine (spec section 10) — the per-commit reliability contract.

The public entry point is :func:`run_gate`, which ties together the pure, LLM-free
pipeline: ``extract → evaluate → raise_floor → choose template``. **No LLM, no network**
in this path (the agent's independent ``assess_escalation`` net is a *separate* layer,
Split 03). The whole pipeline is wrapped **fail-safe**: any exception escalates to a
URGENT caution ("safety check unavailable — please seek in-person care") — it never
swallows an error and returns CLEAR (spec section 18).

**Scope: adult patients (v1)** — pediatric/neonatal red flags are out of scope (Appendix A).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from ..models import (
    EscalationLevel,
    EscalationSource,
    SafetyVerdict,
    Signals,
    TriageBand,
)
from .extractor import extract
from .rules import RULES, evaluate, raise_floor
from .templates import (
    crisis_template,
    emergency_template,
    template_for,
    unavailable_template,
    urgent_template,
)

__all__ = [
    "RULES",
    "GateResult",
    "extract",
    "evaluate",
    "raise_floor",
    "run_gate",
    "emergency_template",
    "crisis_template",
    "urgent_template",
    "unavailable_template",
    "template_for",
]


@dataclass
class GateResult:
    """Outcome of one safety-gate turn (returned by :func:`run_gate` as a dataclass).

    ``failed_safe`` is True only on the fail-safe path (an exception was caught and the
    verdict was forced to a non-CLEAR caution).
    """

    signals: Signals
    verdict: SafetyVerdict
    floor: TriageBand
    template: dict | None
    failed_safe: bool = False

    def as_dict(self) -> dict:
        """JSON-serializable view (Pydantic models dumped)."""
        return {
            "signals": self.signals.model_dump(),
            "verdict": self.verdict.model_dump(),
            "floor": self.floor.value,
            "template": self.template,
            "failed_safe": self.failed_safe,
        }


def run_gate(
    text: str,
    prior_signals: Signals | None = None,
    current_floor: TriageBand = TriageBand.self_care,
    *,
    conn: sqlite3.Connection | None = None,
    session_id: str | None = None,
    msg_id: str | None = None,
) -> GateResult:
    """Run the deterministic safety gate for one patient message.

    Pipeline: ``extract(text, prior) → evaluate(signals) → raise_floor → template``.
    Optionally logs a ``safety_events`` row when given ``conn`` + ``session_id`` (the DB is
    optional so the gate is unit-testable without one).

    **Fail-safe (spec section 18):** any exception in the pipeline returns a non-CLEAR
    URGENT caution verdict and the ``unavailable`` template — never a silent CLEAR.

    Returns:
        A :class:`GateResult`.
    """
    try:
        signals = extract(text, prior_signals)
        verdict = evaluate(signals)
        floor = raise_floor(current_floor, verdict.level)
        template = template_for(verdict.level, verdict.crisis, floor)
    except Exception:
        # Fail safe: escalate to caution, never continue as CLEAR.
        floor = raise_floor(current_floor, EscalationLevel.URGENT)
        verdict = SafetyVerdict(
            level=EscalationLevel.URGENT,
            matched_rules=["safety_check_unavailable"],
            source=EscalationSource.gate,
            crisis=False,
        )
        result = GateResult(
            signals=prior_signals or Signals(),
            verdict=verdict,
            floor=floor,
            template=unavailable_template(floor),
            failed_safe=True,
        )
        _log(conn, session_id, verdict, msg_id)
        return result

    _log(conn, session_id, verdict, msg_id)
    return GateResult(signals=signals, verdict=verdict, floor=floor, template=template)


def _log(
    conn: sqlite3.Connection | None,
    session_id: str | None,
    verdict: SafetyVerdict,
    msg_id: str | None,
) -> None:
    """Best-effort safety_events log; never raises into the caller (audit, not control)."""
    if conn is None or session_id is None:
        return
    try:
        from ..db import log_safety_event

        log_safety_event(
            conn,
            session_id=session_id,
            level=verdict.level.value,
            source=verdict.source.value,
            matched_rules=verdict.matched_rules,
            rules_version=verdict.rules_version,
            msg_id=msg_id,
        )
    except Exception:
        # Logging failure must not break the gate or mask the verdict.
        pass

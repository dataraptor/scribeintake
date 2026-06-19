"""``suggest_triage`` — rule-assisted, floor-clamped triage band (spec §8/§12, Split 04 §3.5).

**Orchestrator-invoked at completion.** The monotonic safety **floor** sets the minimum band;
the model refines within/above it with a one-line rationale. The result is then **clamped in
code** so the predicted band can never drop below the floor — even if the model suggests lower.
That clamp (:func:`clamp_band`) is the deterministic, 100%-gated invariant; it is pure and
unit-tested independent of any model.

Citations stay empty until Split 05 binds retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from ..config import EFFORT_TRIAGE, MAX_TRIAGE_TOKENS
from ..llm import LLMUsage, StructuredClient
from ..models import SOAP, IntakeState, Triage, TriageBand

# Canonical band ordering (definition order of the enum): self_care < gp_routine < gp_urgent < ER.
_BAND_ORDER: list[TriageBand] = list(TriageBand)


class TriageSuggestion(BaseModel):
    """The model's advisory band + rationale (internal call schema, not a cross-module contract)."""

    model_config = ConfigDict(extra="forbid")

    band: TriageBand = TriageBand.self_care
    rationale: str = ""


TRIAGE_SYSTEM = (
    "You are ScribeIntake's triage assistant. Given a SOAP intake summary and a deterministic "
    "safety FLOOR band, choose a triage band and a one-line, non-diagnostic rationale.\n"
    "Bands (increasing urgency): self_care < gp_routine < gp_urgent < ER.\n"
    "RULES:\n"
    "- Never choose a band BELOW the provided floor. You may choose the floor or higher.\n"
    "- Do not diagnose or name a disease; justify by symptoms/red-flags only.\n"
    "- self_care = reassurance + safety-netting; gp_routine = GP in days-weeks; "
    "gp_urgent = GP/urgent care today-soon; ER = emergency now."
)


@dataclass
class TriageResult:
    """Outcome of one ``suggest_triage`` call (clamped triage + usage for the trace)."""

    triage: Triage
    usage: LLMUsage
    model: str
    model_band: TriageBand
    refused: bool = False


def clamp_band(model_band: TriageBand, floor: TriageBand) -> TriageBand:
    """Return the higher of the model's band and the safety floor (never below the floor).

    Pure and deterministic — the project's gated triage invariant. ``max`` over the canonical
    band order, so a model band at or above the floor is kept and a lower one is raised.
    """
    return max(model_band, floor, key=_BAND_ORDER.index)


def suggest_triage(
    state: IntakeState,
    soap: SOAP,
    *,
    floor: TriageBand,
    client: StructuredClient,
    max_tokens: int = MAX_TRIAGE_TOKENS,
) -> TriageResult:
    """Refine the triage band above the floor, then clamp so it is never below the floor."""
    user = (
        f"Safety floor (minimum band): {floor.value}\n\n"
        f"Chief complaint: {soap.subjective.chief_complaint or '(unspecified)'}\n"
        f"Observations: "
        + "; ".join(o.text for o in soap.observations if o.text)
        + "\n\nChoose the triage band and rationale now."
    )
    resp = client.parse(
        system=TRIAGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
        schema=TriageSuggestion,
        effort=EFFORT_TRIAGE,
        max_tokens=max_tokens,
    )

    refused = resp.refused or resp.parsed is None
    suggestion = resp.parsed if not refused else TriageSuggestion(band=floor)
    model_band = suggestion.band  # type: ignore[union-attr]
    band = clamp_band(model_band, floor)  # deterministic floor guarantee
    triage = Triage(band=band, rationale=suggestion.rationale, citations=[])  # type: ignore[union-attr]
    return TriageResult(
        triage=triage,
        usage=resp.usage,
        model=resp.model,
        model_band=model_band,
        refused=refused,
    )

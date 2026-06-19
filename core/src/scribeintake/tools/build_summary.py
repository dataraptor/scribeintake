"""``build_summary`` — the terminal SOAP call (spec §7/§8/§12, Split 04 §3.4).

**Orchestrator-invoked at completion, never agent-chosen.** Produces a schema-valid
:class:`~scribeintake.models.SOAP` from the collected intake slots using **native structured
outputs** (``StructuredClient.parse(schema=SOAP)``) — schema validity is guaranteed at the API
layer, not by a generate-then-validate loop.

The model fills the prose (subjective / objective / observations / a draft triage rationale);
the **deterministic metadata is stamped in code afterwards** so it stays honest and
reproducible regardless of the model: ``low_confidence_fields`` (from recorded confidence),
``red_flags_checked`` (the live rule ids — ``len(RULES)``, never a hardcoded count),
``red_flags_triggered`` (rules that fired on the session signals), ``generated_at`` (passed
in, not a wall clock), and the ``disclaimer`` constant.

Citations stay empty here — Split 05 binds real ``chunk_id``s from ``retrieve_guideline``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import EFFORT_SUMMARY, MAX_SUMMARY_TOKENS
from ..intake import low_confidence_slots
from ..llm import STOP_MAX_TOKENS, LLMUsage, StructuredClient
from ..models import (
    DISCLAIMER,
    SOAP,
    IntakeState,
    Subjective,
)
from ..safety.rules import RULES, evaluate

SUMMARY_SYSTEM = (
    "You are ScribeIntake's summarizer. From the structured intake facts below, produce a "
    "SOAP summary for a CLINICIAN to review before a visit.\n"
    "RULES:\n"
    "- You DO NOT diagnose, name a likely disease, or suggest treatment. Document only.\n"
    "- Fill the OLDCARTS HPI fields and subjective fields ONLY from the provided facts. Do "
    "not invent details. Leave a field empty ('') if the patient did not provide it.\n"
    "- For 'radiation', record where pain spreads (or 'none') if the patient said.\n"
    "- 'objective.patient_reported_vitals' holds ONLY patient-reported numbers; never invent "
    "vitals. 'objective.notes' is brief.\n"
    "- 'observations' are short, non-diagnostic, guideline-style safety-netting notes (e.g. "
    "'Same-day clinician evaluation advised for new exertional chest discomfort'). Leave every "
    "citation field empty ('') — sources are bound by a separate retrieval step.\n"
    "- 'triage.rationale' is a one-line, non-diagnostic justification; leave triage.citations "
    "empty. The band you choose is advisory; a deterministic floor may raise it.\n"
    "- Do not populate 'low_confidence_fields', 'red_flags_checked', 'red_flags_triggered', "
    "'generated_at' or 'disclaimer' — those are filled by code."
)


@dataclass
class SummaryResult:
    """Outcome of one ``build_summary`` call (SOAP + usage for the trace)."""

    soap: SOAP
    usage: LLMUsage
    model: str
    refused: bool = False


def _render_slots(state: IntakeState) -> str:
    """Render the collected slots as labelled lines for the summarizer prompt."""
    lines: list[str] = []
    for key, sv in state.slots.items():
        if not (sv.value and sv.value.strip()):
            continue
        conf = f" (confidence: {sv.confidence.value})" if sv.confidence else ""
        lines.append(f"- {key}: {sv.value}{conf}")
    return "\n".join(lines) if lines else "- (no structured facts recorded)"


def _safe_shell(state: IntakeState) -> SOAP:
    """Minimal, honest SOAP when the model refuses — facts only, no invented prose."""
    cc = state.slots.get("chief_complaint")
    return SOAP(subjective=Subjective(chief_complaint=cc.value if cc else ""))


def build_summary(
    state: IntakeState,
    *,
    client: StructuredClient,
    generated_at: str,
    max_tokens: int = MAX_SUMMARY_TOKENS,
) -> SummaryResult:
    """Generate a schema-valid SOAP from ``state`` (terminal structured-output call).

    On ``stop_reason == "max_tokens"`` the call is retried once with a larger budget; on a
    refusal it falls back to a minimal safe SOAP shell (never crashes). The deterministic
    metadata is always stamped in code after parsing.
    """
    user = (
        "Structured intake facts collected this session:\n"
        f"{_render_slots(state)}\n\n"
        "Produce the SOAP summary now."
    )
    messages = [{"role": "user", "content": user}]

    resp = client.parse(
        system=SUMMARY_SYSTEM,
        messages=messages,
        schema=SOAP,
        effort=EFFORT_SUMMARY,
        max_tokens=max_tokens,
    )
    # Backstop: truncated output -> one retry with a bigger budget (§3.4).
    if resp.stop_reason == STOP_MAX_TOKENS:
        resp = client.parse(
            system=SUMMARY_SYSTEM,
            messages=messages,
            schema=SOAP,
            effort=EFFORT_SUMMARY,
            max_tokens=max_tokens * 2,
        )

    refused = resp.refused or resp.parsed is None
    soap = _safe_shell(state) if refused else resp.parsed  # type: ignore[assignment]

    # --- deterministic metadata (code, not model) --------------------------------
    soap.subjective.low_confidence_fields = low_confidence_slots(state.slots)
    soap.red_flags_checked = [r.id for r in RULES]  # live count, never a hardcoded 18/20
    soap.red_flags_triggered = list(evaluate(state.signals).matched_rules)
    soap.generated_at = generated_at
    soap.disclaimer = DISCLAIMER

    return SummaryResult(soap=soap, usage=resp.usage, model=resp.model, refused=refused)

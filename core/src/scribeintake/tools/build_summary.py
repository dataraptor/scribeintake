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

**Citation binding (Split 05, §12):** after the model writes its (citation-free) observations,
code binds each grounded observation to a real retrieved ``chunk_id``/``url`` via content-term
overlap. A statement no retrieved chunk supports is left **``uncited``** (``citation = None``) —
never a fabricated source. The binding is deterministic (lexical, no model), so it is honest and
unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import (
    CITATION_MIN_OVERLAP,
    CITATION_MIN_SHARED,
    EFFORT_SUMMARY,
    MAX_SUMMARY_TOKENS,
)
from ..intake import low_confidence_slots
from ..llm import STOP_MAX_TOKENS, LLMUsage, StructuredClient
from ..models import (
    DISCLAIMER,
    SOAP,
    Citation,
    IntakeState,
    RetrievedChunk,
    Subjective,
    Triage,
)
from ..rag.text import overlap_score, shared_terms
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
    "'Same-day clinician evaluation advised for new exertional chest discomfort'). When "
    "'Reference guidance' passages are provided below, ground each observation in them and reuse "
    "their wording where appropriate. Leave every citation field empty ('') — sources are bound "
    "by a separate retrieval step in code.\n"
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


def _render_passages(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved guideline passages as 'Reference guidance' for the summarizer prompt."""
    if not chunks:
        return ""
    lines = [f"[{c.source}] {c.text}" for c in chunks]
    return "Reference guidance (public-domain guidelines retrieved for this complaint):\n" + (
        "\n".join(lines) + "\n\n"
    )


def _safe_shell(state: IntakeState) -> SOAP:
    """Minimal, honest SOAP when the model refuses — facts only, no invented prose."""
    cc = state.slots.get("chief_complaint")
    return SOAP(subjective=Subjective(chief_complaint=cc.value if cc else ""))


# ----------------------------------------------------------------- citation binding (§12)
def _best_citation(text: str, chunks: list[RetrievedChunk]) -> Citation | None:
    """Bind ``text`` to its best-supporting chunk, or ``None`` (uncited) if none qualifies.

    A chunk qualifies only when it shares enough content terms with the statement
    (``CITATION_MIN_OVERLAP`` of the statement's terms **and** ``CITATION_MIN_SHARED`` distinct
    terms) — a deliberately conservative bar so generic safety-netting/screening notes ("none
    triggered this session") stay uncited rather than borrow a fabricated source.
    """
    best: RetrievedChunk | None = None
    best_score = 0.0
    for c in chunks:
        score = overlap_score(text, c.text)
        if score > best_score:
            best_score, best = score, c
    if (
        best is not None
        and best_score >= CITATION_MIN_OVERLAP
        and shared_terms(text, best.text) >= CITATION_MIN_SHARED
    ):
        return Citation(source=best.source, url=best.url, chunk_id=best.chunk_id)
    return None


def bind_observation_citations(soap: SOAP, chunks: list[RetrievedChunk]) -> None:
    """Bind each observation to a real chunk in place; unsupported ones become ``uncited``.

    Always normalises: a model-emitted empty/blank citation with no real ``chunk_id`` is reset to
    ``None`` so "uncited" is explicit and a fake ``{source:"", url:"", chunk_id:""}`` never
    persists.
    """
    for obs in soap.observations:
        obs.citation = _best_citation(obs.text, chunks)


def bind_triage_citation(triage: Triage, chunks: list[RetrievedChunk]) -> None:
    """Cite the triage rationale to its best-supporting chunk (e.g. the ER-vs-urgent-care page)."""
    cite = _best_citation(triage.rationale, chunks)
    triage.citations = [cite] if cite is not None else []


def build_summary(
    state: IntakeState,
    *,
    client: StructuredClient,
    generated_at: str,
    chunks: list[RetrievedChunk] | None = None,
    max_tokens: int = MAX_SUMMARY_TOKENS,
) -> SummaryResult:
    """Generate a schema-valid SOAP from ``state`` (terminal structured-output call).

    On ``stop_reason == "max_tokens"`` the call is retried once with a larger budget; on a
    refusal it falls back to a minimal safe SOAP shell (never crashes). The deterministic
    metadata is always stamped in code after parsing.

    ``chunks`` are the retrieved guideline passages; each grounded observation is bound to a real
    ``chunk_id`` and the rest are flagged ``uncited`` (§12). With no chunks (default) every
    observation is normalised to ``uncited`` — never a fabricated source.
    """
    user = (
        "Structured intake facts collected this session:\n"
        f"{_render_slots(state)}\n\n"
        f"{_render_passages(chunks or [])}"
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

    # --- citation binding (code, not model): real chunk_id or `uncited` (§12) -----
    bind_observation_citations(soap, chunks or [])

    return SummaryResult(soap=soap, usage=resp.usage, model=resp.model, refused=refused)

"""Per-turn orchestrator (spec section 6) — the single code path for one patient turn.

:func:`run_turn` implements steps 1–7 in a fixed, **deterministic order**: persist → extract
→ gate → (short-circuit on EMERGENCY) → agent → state update → completion check → persist
traces. The safety guarantee lives in step 3 (code on code-extracted signals), upstream of any
model call: on a gate EMERGENCY the agent **never runs**. The agent's independent
``assess_escalation(EMERGENCY)`` is treated exactly like a gate emergency.

**Stateless per turn:** state is loaded from SQLite at the top and written back at the end;
there is no in-memory session map, which is what keeps eval runs isolated and parallel-safe.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime

from . import db, pricing
from .config import EFFORT_INTAKE, PROMPT_VERSION, RETRIEVE_K, RULES_VERSION, settings
from .intake import compute_branch_hints, compute_open_slots, is_complete
from .llm import StructuredClient
from .models import (
    EscalationLevel,
    EscalationSource,
    IntakeState,
    RetrievedChunk,
    SafetyVerdict,
    ToolCallTrace,
    TriageBand,
)
from .safety import emergency_template, run_gate, urgent_template
from .safety.rules import raise_floor
from .tools import ToolContext
from .tools.build_summary import SummaryResult, bind_triage_citation, build_summary
from .tools.suggest_triage import TriageResult, suggest_triage

logger = logging.getLogger(__name__)

_RANK = {EscalationLevel.CLEAR: 0, EscalationLevel.URGENT: 1, EscalationLevel.EMERGENCY: 2}

_FALLBACK_QUESTION = "Could you tell me a bit more about what's been bothering you?"

_COMPLETION_MESSAGE = (
    "Thanks — that's everything I need for now. I've prepared a summary of what you told me "
    "for your clinician to review. Remember, this is not a diagnosis."
)


@dataclass
class AssistantTurn:
    """Everything one turn produces — shaped to feed the frontend inline strip + the API.

    Carries the assistant text, the safety verdict (level/source/rules/crisis), the resulting
    monotonic floor and status, the optional safety ``template`` dict, a signals snapshot, the
    open slots, the tools the agent used, and the trace rows written this turn.
    """

    session_id: str
    turn: int
    assistant_text: str
    level: EscalationLevel
    source: EscalationSource
    matched_rules: list[str]
    crisis: bool
    triage_floor: TriageBand
    status: str
    template: dict | None
    signals: dict
    open_slots: list[str]
    tools_used: list[str] = field(default_factory=list)
    traces: list[ToolCallTrace] = field(default_factory=list)
    model: str | None = None
    failed_safe: bool = False
    # Populated only on the completion turn (the finalized SOAP + clamped predicted band).
    soap: dict | None = None
    triage_band: TriageBand | None = None

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turn": self.turn,
            "assistant_text": self.assistant_text,
            "level": self.level.value,
            "source": self.source.value,
            "matched_rules": self.matched_rules,
            "crisis": self.crisis,
            "triage_floor": self.triage_floor.value,
            "status": self.status,
            "template": self.template,
            "signals": self.signals,
            "open_slots": self.open_slots,
            "tools_used": self.tools_used,
            "model": self.model,
            "failed_safe": self.failed_safe,
            "soap": self.soap,
            "triage_band": self.triage_band.value if self.triage_band else None,
        }


def run_turn(
    session_id: str,
    user_msg: str,
    *,
    conn: sqlite3.Connection,
    agent: object | None = None,
    summary_client: StructuredClient | None = None,
    retriever: object | None = None,
    generated_at: str | None = None,
    effort: str = EFFORT_INTAKE,
) -> AssistantTurn:
    """Run one patient turn end-to-end (steps 1–7, §6).

    ``agent`` is an :class:`~scribeintake.agent.AgentLoop` (or a compatible double for tests);
    if ``None`` the live agent is built lazily — but only when the turn actually reaches the
    model step, so the EMERGENCY short-circuit needs no credentials. ``summary_client`` is the
    :class:`~scribeintake.llm.StructuredClient` for the terminal summary/triage calls (lazily
    built only when a turn actually completes). ``retriever`` is the RAG
    :class:`~scribeintake.rag.HybridRetriever` used for ``retrieve_guideline`` + citation
    binding; when ``None`` it is lazily loaded **only on the full production path** (i.e. when
    ``agent`` is also lazily built), so the deterministic tier — which injects an ``agent`` —
    never loads an index. A missing/unbuilt index degrades gracefully to ``uncited``.
    ``generated_at`` is the ISO timestamp stamped into the SOAP — passed in so eval/cached paths
    stay reproducible (§3.4).
    """
    # --- load state (stateless-per-turn) -----------------------------------------
    state = db.load_intake_state(conn, session_id)

    # --- step 1: persist the raw user message ------------------------------------
    msg_id = db.add_message(conn, session_id, "user", user_msg)
    turn = db.count_user_messages(conn, session_id)

    # --- steps 2-3: extract + gate (code, NO LLM) --------------------------------
    gate = run_gate(
        user_msg,
        prior_signals=state.signals,
        current_floor=state.triage_floor,
        conn=conn,
        session_id=session_id,
        msg_id=str(msg_id),
    )
    state.signals = gate.signals
    state.triage_floor = gate.floor
    if state.triage_floor is not TriageBand.self_care:
        state.floor_pinned = True

    # --- step 3 short-circuit: gate EMERGENCY → template + halt, agent NEVER runs -
    if gate.verdict.level is EscalationLevel.EMERGENCY:
        return _halt(
            conn,
            state,
            turn,
            template=gate.template,
            verdict=gate.verdict,
            failed_safe=gate.failed_safe,
        )

    # --- step 4: agent (CLEAR or URGENT continue) --------------------------------
    # The production path builds both the agent and the retriever lazily; an injected agent
    # (the deterministic tier) suppresses the retriever auto-load so no index is required.
    lazy_path = agent is None
    if agent is None:
        from .agent import build_default_agent

        agent = build_default_agent()
    if retriever is None and lazy_path:
        retriever = _lazy_retriever()

    ctx = ToolContext(
        session_id=session_id,
        turn=turn,
        state=state,
        conn=conn,
        rules_version=RULES_VERSION,
        msg_id=str(msg_id),
        retriever=retriever,
    )
    ctx.open_slots = compute_open_slots(state.slots)
    ctx.branch_hints = compute_branch_hints(state.slots)

    history = _history_messages(conn, session_id, before_id=msg_id)
    reminder = (
        f"<system-reminder>open_slots={ctx.open_slots}; "
        f"branch_hints={ctx.branch_hints}</system-reminder>"
    )
    user_content = [
        {"type": "text", "text": user_msg},
        {"type": "text", "text": reminder},
    ]

    result = agent.run_turn(history=history, user_content=user_content, ctx=ctx, effort=effort)
    traces = _persist_traces(conn, session_id, turn, result)
    model_used = _last_model(result)

    # --- agent EMERGENCY → treat exactly like a gate emergency (discard question) -
    if ctx.agent_escalation is EscalationLevel.EMERGENCY:
        state.triage_floor = raise_floor(state.triage_floor, EscalationLevel.EMERGENCY)
        state.floor_pinned = True
        verdict = SafetyVerdict(
            level=EscalationLevel.EMERGENCY,
            matched_rules=["agent_assessment"],
            source=EscalationSource.agent,
            crisis=False,
        )
        return _halt(
            conn,
            state,
            turn,
            template=emergency_template(),
            verdict=verdict,
            traces=traces,
            tools_used=result.tools_used,
            model=model_used,
        )

    # --- combine gate + agent verdicts (escalate-only) ---------------------------
    level = gate.verdict.level
    source = gate.verdict.source
    matched = list(gate.verdict.matched_rules)
    crisis = gate.verdict.crisis
    template = gate.template

    if ctx.agent_escalation is EscalationLevel.URGENT:
        state.triage_floor = raise_floor(state.triage_floor, EscalationLevel.URGENT)
        state.floor_pinned = True
        if _RANK[EscalationLevel.URGENT] > _RANK[level]:
            level = EscalationLevel.URGENT
            source = EscalationSource.agent
            matched = ["agent_assessment"]
            template = template or urgent_template(state.triage_floor)

    # --- step 6: completion check (orchestrator-owned, deterministic) -------------
    open_slots = compute_open_slots(state.slots)
    soap_dict: dict | None = None
    triage_band: TriageBand | None = None

    if is_complete(state.slots, turn):
        # --- finalization: build_summary -> suggest_triage -> persist (§3.6) ------
        soap_dict, triage_band, fin_traces = _finalize(
            conn,
            state,
            turn,
            summary_client=summary_client,
            retriever=retriever,
            generated_at=generated_at,
        )
        traces.extend(fin_traces)
        state.status = "completed"
        assistant_text = _COMPLETION_MESSAGE
    else:
        state.status = "active"
        assistant_text = result.assistant_text or _FALLBACK_QUESTION

    # --- step 7: persist assistant message + state -------------------------------
    db.add_message(conn, session_id, "assistant", assistant_text, model=model_used)
    db.save_intake_state(conn, state)
    if triage_band is not None:
        # completed: persist the summary row + final band (after state, which sets status).
        db.save_summary(conn, session_id, json.dumps(soap_dict), PROMPT_VERSION)
        db.finalize_session(conn, session_id, triage_band.value)

    return AssistantTurn(
        session_id=session_id,
        turn=turn,
        assistant_text=assistant_text,
        level=level,
        source=source,
        matched_rules=matched,
        crisis=crisis,
        triage_floor=state.triage_floor,
        status=state.status,
        template=template,
        signals=state.signals.model_dump(),
        open_slots=open_slots,
        tools_used=result.tools_used,
        traces=traces,
        model=model_used,
        failed_safe=gate.failed_safe,
        soap=soap_dict,
        triage_band=triage_band,
    )


# ----------------------------------------------------------------------- helpers
def _finalize(
    conn: sqlite3.Connection,
    state: IntakeState,
    turn: int,
    *,
    summary_client: StructuredClient | None,
    retriever: object | None,
    generated_at: str | None,
) -> tuple[dict, TriageBand, list[ToolCallTrace]]:
    """Run the two terminal Opus→GPT-5.5 calls and return ``(soap_dict, band, traces)``.

    ``build_summary`` (schema-valid SOAP, with retrieved chunks bound as citations) then
    ``suggest_triage`` (band clamped ≥ floor). The clamped band is written into the SOAP's
    ``triage`` block — and its rationale cited — so the persisted summary and the session band
    agree. The summary client is built lazily here — only a turn that actually completes needs
    credentials.
    """
    if summary_client is None:
        from .llm import build_summary_client

        summary_client = build_summary_client(settings)
    stamp = generated_at or datetime.now(UTC).isoformat()

    chunks = _retrieve_for_summary(state, retriever)
    summary = build_summary(state, client=summary_client, generated_at=stamp, chunks=chunks)
    triage = suggest_triage(state, summary.soap, floor=state.triage_floor, client=summary_client)
    summary.soap.triage = triage.triage  # clamped band into the SOAP
    bind_triage_citation(summary.soap.triage, chunks)  # cite the rationale (or leave empty)

    traces = _persist_finalize_traces(conn, state.session_id, turn, summary, triage)
    return summary.soap.model_dump(), triage.triage.band, traces


def _summary_query(state: IntakeState) -> str:
    """Build a retrieval query from the chief complaint + salient HPI for citation binding."""
    parts: list[str] = []
    for key in ("chief_complaint", "hpi.character", "hpi.radiation", "hpi.location"):
        sv = state.slots.get(key)
        if sv and sv.value and sv.value.strip().lower() not in ("", "none", "unknown"):
            parts.append(sv.value.strip())
    base = " ".join(parts) or "general symptoms"
    return f"{base} when to seek emergency care"


def _retrieve_for_summary(state: IntakeState, retriever: object | None) -> list[RetrievedChunk]:
    """Best-effort retrieval for citation binding; empty (→ uncited) on any failure (§18)."""
    if retriever is None:
        return []
    try:
        return retriever.retrieve(_summary_query(state), k=RETRIEVE_K)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - never fail finalization on retrieval
        logger.warning("summary retrieval failed, proceeding uncited: %s", exc)
        return []


def _lazy_retriever() -> object | None:
    """Load the process-cached live retriever; ``None`` if no index is built yet (§18)."""
    try:
        from .rag import get_retriever

        return get_retriever()
    except Exception as exc:  # noqa: BLE001 - unbuilt/unreadable index → degrade to uncited
        logger.warning("RAG retriever unavailable, proceeding without citations: %s", exc)
        return None


def _persist_finalize_traces(
    conn: sqlite3.Connection,
    session_id: str,
    turn: int,
    summary: SummaryResult,
    triage: TriageResult,
) -> list[ToolCallTrace]:
    """Write one ``tool_calls`` row per terminal structured-output call (with cost)."""
    traces: list[ToolCallTrace] = []
    for tool, model, usage, result_json in (
        ("build_summary", summary.model, summary.usage, {"refused": summary.refused}),
        (
            "suggest_triage",
            triage.model,
            triage.usage,
            {"model_band": triage.model_band.value, "band": triage.triage.band.value},
        ),
    ):
        cost = (
            pricing.cost_usd(
                model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_creation_tokens,
                usage.cache_read_tokens,
            )
            if model in pricing.PRICES
            else 0.0
        )
        tr = ToolCallTrace(
            session_id=session_id,
            turn=turn,
            tool=tool,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_creation_tokens=usage.cache_creation_tokens,
            cost_usd=cost,
            result_json=json.dumps(result_json),
        )
        db.log_tool_call(conn, tr)
        traces.append(tr)
    return traces


def _halt(
    conn: sqlite3.Connection,
    state: IntakeState,
    turn: int,
    *,
    template: dict | None,
    verdict: SafetyVerdict,
    traces: list[ToolCallTrace] | None = None,
    tools_used: list[str] | None = None,
    model: str | None = None,
    failed_safe: bool = False,
) -> AssistantTurn:
    """Emit a safety template, set status=halted, persist, and return the turn."""
    text = _template_text(template)
    state.status = "halted"
    db.add_message(conn, state.session_id, "assistant", text, model=model)
    db.save_intake_state(conn, state)
    return AssistantTurn(
        session_id=state.session_id,
        turn=turn,
        assistant_text=text,
        level=verdict.level,
        source=verdict.source,
        matched_rules=list(verdict.matched_rules),
        crisis=verdict.crisis,
        triage_floor=state.triage_floor,
        status=state.status,
        template=template,
        signals=state.signals.model_dump(),
        open_slots=compute_open_slots(state.slots),
        tools_used=tools_used or [],
        traces=traces or [],
        model=model,
        failed_safe=failed_safe,
    )


def _history_messages(
    conn: sqlite3.Connection,
    session_id: str,
    before_id: int,
) -> list[dict]:
    """Rebuild OpenAI-shape conversation history (prior messages only)."""
    rows = db.get_messages(conn, session_id, before_id=before_id)
    out: list[dict] = []
    for r in rows:
        role = "assistant" if r["role"] == "assistant" else "user"
        out.append({"role": role, "content": r["content"]})
    return out


def _persist_traces(
    conn: sqlite3.Connection,
    session_id: str,
    turn: int,
    result: object,
) -> list[ToolCallTrace]:
    """Write one ``tool_calls`` row per model call and per local tool execution."""
    traces: list[ToolCallTrace] = []
    for step in result.steps:  # type: ignore[attr-defined]
        cost = (
            pricing.cost_usd(
                step.model,
                step.usage.input_tokens,
                step.usage.output_tokens,
                step.usage.cache_creation_tokens,
                step.usage.cache_read_tokens,
            )
            if step.model in pricing.PRICES
            else 0.0
        )
        tr = ToolCallTrace(
            session_id=session_id,
            turn=turn,
            tool="agent_step",
            model=step.model,
            input_tokens=step.usage.input_tokens,
            output_tokens=step.usage.output_tokens,
            cache_read_tokens=step.usage.cache_read_tokens,
            cache_creation_tokens=step.usage.cache_creation_tokens,
            latency_ms=step.latency_ms,
            cost_usd=cost,
            result_json=json.dumps(
                {"text": step.text[:500], "tool_calls": [tc.name for tc in step.tool_calls]}
            ),
        )
        db.log_tool_call(conn, tr)
        traces.append(tr)

    for ex in result.tool_executions:  # type: ignore[attr-defined]
        tr = ToolCallTrace(
            session_id=session_id,
            turn=turn,
            tool=ex.name,
            model=None,
            cost_usd=0.0,
            latency_ms=ex.latency_ms,
            args_json=json.dumps(ex.arguments),
            result_json=json.dumps(ex.result),
        )
        db.log_tool_call(conn, tr)
        traces.append(tr)

    return traces


def _last_model(result: object) -> str | None:
    steps = result.steps  # type: ignore[attr-defined]
    return steps[-1].model if steps else None


def _template_text(template: dict | None) -> str:
    if not template:
        return ""
    parts = [template.get("heading", ""), template.get("body", "")]
    return "\n\n".join(p for p in parts if p)

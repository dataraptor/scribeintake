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
import sqlite3
from dataclasses import dataclass, field

from . import db, pricing
from .config import EFFORT_INTAKE, MAX_INTAKE_TURNS, RULES_VERSION
from .intake import compute_branch_hints, compute_open_slots
from .models import (
    EscalationLevel,
    EscalationSource,
    IntakeState,
    SafetyVerdict,
    ToolCallTrace,
    TriageBand,
)
from .safety import emergency_template, run_gate, urgent_template
from .safety.rules import raise_floor
from .tools import ToolContext

_RANK = {EscalationLevel.CLEAR: 0, EscalationLevel.URGENT: 1, EscalationLevel.EMERGENCY: 2}

_FALLBACK_QUESTION = "Could you tell me a bit more about what's been bothering you?"


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
        }


def run_turn(
    session_id: str,
    user_msg: str,
    *,
    conn: sqlite3.Connection,
    agent: object | None = None,
    effort: str = EFFORT_INTAKE,
) -> AssistantTurn:
    """Run one patient turn end-to-end (steps 1–7, §6).

    ``agent`` is an :class:`~scribeintake.agent.AgentLoop` (or a compatible double for tests);
    if ``None`` the live agent is built lazily — but only when the turn actually reaches the
    model step, so the EMERGENCY short-circuit needs no credentials.
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
    if agent is None:
        from .agent import build_default_agent

        agent = build_default_agent()

    ctx = ToolContext(
        session_id=session_id,
        turn=turn,
        state=state,
        conn=conn,
        rules_version=RULES_VERSION,
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
    if not open_slots or turn >= MAX_INTAKE_TURNS:
        state.status = "ready_to_summarize"
    else:
        state.status = "active"

    # --- step 7: persist assistant message + state -------------------------------
    assistant_text = result.assistant_text or _FALLBACK_QUESTION
    db.add_message(conn, session_id, "assistant", assistant_text, model=model_used)
    db.save_intake_state(conn, state)

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
    )


# ----------------------------------------------------------------------- helpers
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

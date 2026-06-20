"""Pure mappers: engine objects -> HTTP DTOs (Split 10, spec §14).

These functions take an :class:`~scribeintake.orchestrator.AssistantTurn`, a persisted SOAP dict,
or ``tool_calls`` rows and produce the :mod:`api.schemas` DTOs whose field names match the
frontend view-model. They are **pure** (no DB, no network, no model call) so they unit-test
without HTTP, and they keep ``main.py`` thin. No safety/intake logic lives here — the verdict,
templates and SOAP are already produced by ``core`` upstream; this module only reshapes them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from scribeintake import pricing
from scribeintake.models import DISCLAIMER, EscalationLevel, EscalationSource, TriageBand

from . import schemas

# The summary's subjective rows the mockup shows, in display order (mockup ``openSummary`` FIELDS).
_SUBJECTIVE_FIELDS: list[tuple[str, str]] = [
    ("chief_complaint", "subjective.chief_complaint"),
    ("hpi.onset", "subjective.hpi.onset"),
    ("hpi.severity", "subjective.hpi.severity"),
    ("hpi.radiation", "subjective.hpi.radiation"),
    ("medications", "subjective.medications"),
    ("allergies", "subjective.allergies"),
]

# Vitals shown on the objective one-liner, in order.
_VITAL_LABELS: list[tuple[str, str]] = [
    ("glucose", "glucose_mgdl"),
    ("SpO2", "spo2"),
    ("HR", "hr"),
    ("temp", "temp_f"),
]


# --------------------------------------------------------------------------------- turn mapping
def turn_response(turn: Any) -> schemas.TurnResponse:
    """Map an :class:`AssistantTurn` to a :class:`~api.schemas.TurnResponse`."""
    is_emergency = turn.level is EscalationLevel.EMERGENCY and turn.template
    emergency = emergency_payload(turn) if is_emergency else None
    return schemas.TurnResponse(
        session_id=turn.session_id,
        turn=turn.turn,
        content=turn.assistant_text,
        model=turn.model,
        sources=_sources_from_traces(turn.traces),
        level=turn.level.value,
        source=turn.source.value,
        status=turn.status,
        crisis=turn.crisis,
        triage_floor=turn.triage_floor.value,
        floor_pinned=turn.triage_floor is not TriageBand.self_care,
        ready_to_summarize=turn.status == "completed",
        triage_band=turn.triage_band.value if turn.triage_band else None,
        strip=strip_view(turn),
        emergency=emergency,
        trace_delta=[_trace_row(t) for t in turn.traces],
        disclaimer=DISCLAIMER,
    )


def strip_view(turn: Any) -> schemas.StripView:
    """Build the inline safety strip from the turn verdict."""
    agent_net = turn.source is EscalationSource.agent
    return schemas.StripView(
        level=turn.level.value,
        source=turn.source.value,
        agent_net=agent_net,
        crisis=turn.crisis,
        rule_id=_rule_id(turn),
        rule_level=turn.level.value,
        rule_source=turn.source.value,
        signals=dict(turn.signals or {}),
        signals_view=_signals_view(turn.signals or {}),
        tools=list(turn.tools_used),
        tools_note=_tools_note(turn),
        model=turn.model,
    )


def emergency_payload(turn: Any) -> schemas.EmergencyPayload:
    """Map the core safety template dict to the emergency sheet payload (wording verbatim)."""
    t: Mapping[str, Any] = turn.template or {}
    kind = t.get("kind", "emergency")
    note = t.get("note", "")
    return schemas.EmergencyPayload(
        kind=kind,
        crisis=kind == "crisis" or bool(turn.crisis),
        kicker=t.get("kicker", ""),
        heading=t.get("heading", ""),
        body=t.get("body", ""),
        note=note,
        has_note=bool(note),
        actions=[
            schemas.Action(label=a.get("label", ""), href=a.get("href", ""))
            for a in t.get("actions", [])
        ],
        caption=_caption(turn),
        disclaimer=t.get("disclaimer", DISCLAIMER),
    )


def _rule_id(turn: Any) -> str:
    if turn.matched_rules:
        return turn.matched_rules[0]
    return "no rule matched" if turn.level is EscalationLevel.CLEAR else "—"


def _tools_note(turn: Any) -> str:
    """The strip's short provenance note on an emergency short-circuit (presentation only)."""
    if turn.level is not EscalationLevel.EMERGENCY:
        return ""
    if turn.source is EscalationSource.agent:
        return "assess_escalation → EMERGENCY (regex missed it)"
    return "gate short-circuited — agent never ran"


def _caption(turn: Any) -> str:
    """A provenance caption for the sheet, derived from the verdict (not safety wording)."""
    if turn.crisis:
        return "Crisis template · compassionate tone · logged separately · no model call"
    if turn.source is EscalationSource.agent:
        return (
            "AI assessment · 2nd net · assess_escalation → EMERGENCY · "
            "the regex layer missed this phrasing"
        )
    rule = turn.matched_rules[0] if turn.matched_rules else "gate"
    return f"Deterministic rule · {rule} · no model call · 0 missed on frozen set"


def _signals_view(signals: Mapping[str, Any]) -> list[schemas.SignalView]:
    """The present/true signals as ``{name, mark}`` (False booleans and absent numerics dropped)."""
    out: list[schemas.SignalView] = []
    for name, val in signals.items():
        if isinstance(val, bool):
            if val:
                out.append(schemas.SignalView(name=name, mark="✓"))
        elif val is not None:
            out.append(schemas.SignalView(name=name, mark=str(val)))
    return out


def _sources_from_traces(traces: Iterable[Any]) -> list[schemas.Source]:
    """Surface retrieved guideline sources from this turn's ``retrieve_guideline`` trace rows."""
    out: list[schemas.Source] = []
    seen: set[str] = set()
    for tr in traces:
        if getattr(tr, "tool", None) != "retrieve_guideline":
            continue
        chunks = _result_chunks(getattr(tr, "result_json", None))
        for c in chunks:
            cid = c.get("chunk_id", "")
            if cid and cid in seen:
                continue
            seen.add(cid)
            out.append(
                schemas.Source(
                    source=c.get("source", ""),
                    chunk_id=cid,
                    url=c.get("url", ""),
                    score=float(c.get("score", 0.0) or 0.0),
                )
            )
    return out


def _result_chunks(result_json: Any) -> list[dict]:
    import json

    if not result_json:
        return []
    try:
        data = json.loads(result_json) if isinstance(result_json, str) else result_json
    except (ValueError, TypeError):
        return []
    chunks = data.get("chunks") if isinstance(data, dict) else None
    return [c for c in chunks if isinstance(c, dict)] if isinstance(chunks, list) else []


def _trace_row(t: Any) -> schemas.TraceRowView:
    """Map a :class:`ToolCallTrace` (turn delta) to a trace row view."""
    model = getattr(t, "model", None)
    tool = getattr(t, "tool", "")
    return schemas.TraceRowView(
        tool=tool,
        model=model,
        latency_ms=getattr(t, "latency_ms", None),
        cost_usd=float(getattr(t, "cost_usd", 0.0) or 0.0),
        local=model is None,
        event=str(tool).startswith("safety_event"),
    )


# ------------------------------------------------------------------------------ summary mapping
def summary_response(soap: dict, *, band: str | None = None) -> schemas.SummaryResponse:
    """Map a persisted SOAP dict (+ the session's clamped band) to a summary payload."""
    subj = soap.get("subjective", {}) or {}
    low = list(subj.get("low_confidence_fields", []) or [])
    fields: list[schemas.SubjectiveField] = []
    for key, path in _SUBJECTIVE_FIELDS:
        value = _get_path(soap, path)
        value_str = ", ".join(str(v) for v in value) if isinstance(value, list) else (value or "")
        fields.append(schemas.SubjectiveField(key=key, value=value_str, low=key in low))

    triage = soap.get("triage", {}) or {}
    band_val = band or triage.get("band", TriageBand.self_care.value)
    return schemas.SummaryResponse(
        band=band_val,
        subjective=fields,
        objective=_objective_line(soap.get("objective", {}) or {}),
        observations=[_observation_view(o) for o in soap.get("observations", []) or []],
        low_confidence_fields=low,
        red_flags_checked=len(soap.get("red_flags_checked", []) or []),
        red_flags_triggered=len(soap.get("red_flags_triggered", []) or []),
        generated_at=soap.get("generated_at", ""),
        disclaimer=soap.get("disclaimer", DISCLAIMER),
        soap=soap,
    )


def _observation_view(o: Mapping[str, Any]) -> schemas.ObservationView:
    cit = o.get("citation") or {}
    chunk = cit.get("chunk_id", "") if isinstance(cit, Mapping) else ""
    if chunk:
        return schemas.ObservationView(
            text=o.get("text", ""),
            cited=True,
            uncited=False,
            source=cit.get("source", ""),
            chunk=chunk,
            url=cit.get("url", ""),
        )
    return schemas.ObservationView(text=o.get("text", ""), cited=False, uncited=True)


def _objective_line(obj: Mapping[str, Any]) -> str:
    """A short patient-reported-vitals one-liner (mockup ``objective``)."""
    vit = obj.get("patient_reported_vitals", {}) or {}
    parts: list[str] = []
    sbp, dbp = vit.get("sbp"), vit.get("dbp")
    if sbp and dbp:
        parts.append(f"BP {sbp}/{dbp} (home)")
    elif sbp:
        parts.append(f"SBP {sbp} (home)")
    for label, key in _VITAL_LABELS:
        v = vit.get(key)
        if v:
            parts.append(f"{label} {v}")
    notes = obj.get("notes", "") or ""
    if not parts:
        return notes or "none reported"
    line = ", ".join(parts)
    return f"{line}; {notes}" if notes else line


def _get_path(soap: Mapping[str, Any], path: str) -> Any:
    cur: Any = soap
    for part in path.split("."):
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(part)
    return cur


# -------------------------------------------------------------------------------- trace mapping
def trace_response(rows: Iterable[Any], session_id: str) -> schemas.TraceResponse:
    """Map a session's ``tool_calls`` rows to a trace payload with cache-aware totals.

    Totals are recomputed from the persisted token buckets via ``pricing.savings_vs_no_cache`` so
    the cost and the "cache N% saved" label are honest (local $0 rows contribute nothing).
    """
    rows = list(rows)
    views = [
        schemas.TraceRowView(
            tool=r["tool"],
            model=r["model"],
            latency_ms=r["latency_ms"],
            cost_usd=float(r["cost_usd"] or 0.0),
            local=r["model"] is None,
            event=str(r["tool"]).startswith("safety_event"),
        )
        for r in rows
    ]
    buckets = [
        {
            "model": r["model"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "cache_creation_tokens": r["cache_creation_tokens"],
            "cache_read_tokens": r["cache_read_tokens"],
        }
        for r in rows
    ]
    sav = pricing.savings_vs_no_cache(buckets)
    total = sav["with_cache"]
    pct = sav["pct_saved"]
    n_turns = max((int(r["turn"] or 0) for r in rows), default=0)
    label = f"${total:.4f} · cache {round(pct * 100)}% saved"
    return schemas.TraceResponse(
        session_id=session_id,
        rows=views,
        n_turns=n_turns,
        total_cost_usd=total,
        pct_cache_saved=pct,
        trace_cost_label=label,
    )

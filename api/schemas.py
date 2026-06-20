"""Request/response DTOs for the HTTP API (Split 10, spec §14).

These models mirror the orchestrator's :class:`~scribeintake.orchestrator.AssistantTurn` /
:class:`~scribeintake.models.SOAP` / trace rows and **map onto the frontend's existing
view-model field names** (the ``.dc.html`` component: ``stripVM`` / ``applyTurn`` /
``toEmergency`` / ``openSummary`` / the trace rows), so Split 11 is a data-layer swap with no
UI reshape. Where the mockup uses camelCase (the inline-strip data: ``ruleId`` / ``ruleLevel`` /
``ruleSource`` / ``signalsView`` / ``toolsNote``; ``agentNet``; ``hasNote``; ``chunkId``) the
field carries a camelCase **alias** so the JSON keys match exactly — FastAPI serialises responses
by alias by default. Top-level fields stay snake_case (clean API convention); a couple carry the
mockup's camelCase alias (``sessionId`` / ``triageFloor`` / ``floorPinned`` / ``readyToSummarize``
/ ``triageBand`` / ``traceDelta``) for the same identity-mapping reason.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from scribeintake.models import DISCLAIMER


class _Base(BaseModel):
    # populate_by_name lets serialize.py construct with snake_case while JSON emits the alias.
    model_config = ConfigDict(populate_by_name=True)


# --------------------------------------------------------------------------- message sources
class Source(_Base):
    """One retrieved guideline source shown under an assistant message (mockup ``sources[]``)."""

    source: str = ""
    chunk_id: str = Field("", alias="chunkId")
    url: str = ""
    score: float = 0.0


# ------------------------------------------------------------------------- inline safety strip
class SignalView(_Base):
    """One extracted signal for the strip's signals view (a present/true signal)."""

    name: str
    mark: str


class StripView(_Base):
    """The per-turn inline safety strip (mockup ``stripVM`` inputs)."""

    level: str
    source: str
    agent_net: bool = Field(False, alias="agentNet")
    crisis: bool = False
    rule_id: str = Field("—", alias="ruleId")
    rule_level: str = Field("", alias="ruleLevel")
    rule_source: str = Field("", alias="ruleSource")
    # Raw extracted signals (the client styles them); signals_view is a ready-made list of the
    # present/true signals for a no-compute render. Either may be used by Split 11.
    signals: dict = Field(default_factory=dict)
    signals_view: list[SignalView] = Field(default_factory=list, alias="signalsView")
    tools: list[str] = Field(default_factory=list)
    tools_note: str = Field("", alias="toolsNote")
    model: str | None = None


# ------------------------------------------------------------------------------ emergency sheet
class Action(_Base):
    """A call-to-action button (a ``tel:`` link) on a safety sheet."""

    label: str
    href: str


class EmergencyPayload(_Base):
    """The emergency/crisis sheet (frontend ``toEmergency``), built from the core templates.

    The safety wording (``kicker`` / ``heading`` / ``body`` / ``actions``) is copied verbatim
    from ``core/safety/templates.py`` — never re-authored here. Only ``caption`` (a provenance
    label) is derived from the verdict.
    """

    kind: str
    crisis: bool = False
    kicker: str = ""
    heading: str = ""
    body: str = ""
    note: str = ""
    has_note: bool = Field(False, alias="hasNote")
    actions: list[Action] = Field(default_factory=list)
    caption: str = ""
    disclaimer: str = DISCLAIMER


# -------------------------------------------------------------------------------------- trace
class TraceRowView(_Base):
    """One trace row (mockup proof-tab trace): a model call or a local ($0) tool/event row."""

    tool: str
    model: str | None = None
    latency_ms: int | None = Field(None, alias="latencyMs")
    cost_usd: float = Field(0.0, alias="costUsd")
    local: bool = False
    event: bool = False


# ---------------------------------------------------------------------------- the turn response
class TurnResponse(_Base):
    """Everything one patient turn produces over HTTP (mockup message + strip + sheet + trace)."""

    session_id: str = Field(..., alias="sessionId")
    turn: int
    content: str
    model: str | None = None
    sources: list[Source] = Field(default_factory=list)
    level: str
    source: str
    status: str
    crisis: bool = False
    triage_floor: str = Field(..., alias="triageFloor")
    floor_pinned: bool = Field(False, alias="floorPinned")
    ready_to_summarize: bool = Field(False, alias="readyToSummarize")
    triage_band: str | None = Field(None, alias="triageBand")
    # Canonical slot keys still needing an answer (the engine's open required/branch slots). The
    # frontend marks a slot filled when it is *not* in this list, so the intake progress bar is
    # honest without a per-slot value contract. Empty once intake is complete.
    open_slots: list[str] = Field(default_factory=list, alias="openSlots")
    strip: StripView
    emergency: EmergencyPayload | None = None
    trace_delta: list[TraceRowView] = Field(default_factory=list, alias="traceDelta")
    disclaimer: str = DISCLAIMER


# ------------------------------------------------------------------------------------ summary
class SubjectiveField(_Base):
    """One summary subjective row (mockup ``openSummary`` ``subjective[]``)."""

    key: str
    value: str
    low: bool = False


class ObservationView(_Base):
    """One summary observation with its citation state (mockup ``observations[]``)."""

    text: str
    cited: bool = False
    uncited: bool = True
    source: str = ""
    chunk: str = ""
    url: str = ""


class SummaryResponse(_Base):
    """The persisted SOAP, shaped for the summary sheet. Always carries the disclaimer (§14)."""

    band: str
    subjective: list[SubjectiveField] = Field(default_factory=list)
    objective: str = ""
    observations: list[ObservationView] = Field(default_factory=list)
    low_confidence_fields: list[str] = Field(default_factory=list)
    red_flags_checked: int = 0
    red_flags_triggered: int = 0
    generated_at: str = ""
    disclaimer: str = DISCLAIMER
    # The full structured SOAP for any client that wants the raw subtree.
    soap: dict = Field(default_factory=dict)


class TraceResponse(_Base):
    """The session trace + totals (mockup proof tab: rows + ``traceCost`` label)."""

    session_id: str = Field(..., alias="sessionId")
    rows: list[TraceRowView] = Field(default_factory=list)
    n_turns: int = Field(0, alias="nTurns")
    total_cost_usd: float = Field(0.0, alias="totalCostUsd")
    pct_cache_saved: float = Field(0.0, alias="pctCacheSaved")
    trace_cost_label: str = Field("", alias="traceCostLabel")


# ----------------------------------------------------------------------- session / health / err
class StartSessionResponse(_Base):
    session_id: str = Field(..., alias="sessionId")
    disclaimer: str = DISCLAIMER


class MessageRequest(_Base):
    text: str


class HealthResponse(_Base):
    status: str = "ok"
    version: str
    models: dict


class ErrorResponse(_Base):
    """A friendly error payload — never a blank failure (§18)."""

    error: str
    detail: str = ""

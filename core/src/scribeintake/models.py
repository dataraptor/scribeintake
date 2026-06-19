"""Pydantic v2 schemas — the cross-module contracts.

Field names mirror spec section 12 / Appendix A and the ``.dc.html`` mockup so the
frontend (Split 11) binds without drift.

The SOAP subtree is **native-structured-output-safe** (spec section 12): every object
sets ``extra="forbid"`` (``additionalProperties: false``), uses enums for fixed sets,
carries no ``minLength``/``maxLength``/``maximum``/``pattern`` constraints, and is
non-recursive. This is asserted in ``tests/test_models.py`` so Split 04 can hand the
schema straight to ``output_config.format``.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .config import PROMPT_VERSION, RULES_VERSION

DISCLAIMER = "Not a diagnosis. For clinician review."


# --------------------------------------------------------------------------- enums
class Confidence(StrEnum):
    high = "high"
    medium = "medium"
    unknown = "unknown"


class EscalationLevel(StrEnum):
    CLEAR = "CLEAR"
    URGENT = "URGENT"
    EMERGENCY = "EMERGENCY"


class TriageBand(StrEnum):
    self_care = "self_care"
    gp_routine = "gp_routine"
    gp_urgent = "gp_urgent"
    ER = "ER"


class EscalationSource(StrEnum):
    gate = "gate"
    agent = "agent"


# ------------------------------------------------------------------------- signals
class Signals(BaseModel):
    """Extractor output (spec Appendix A) — the safety gate's only input.

    Symptom booleans default ``False``; numerics default ``None`` (absent). Split 02
    fills these from regex/number parsing of the raw message plus typed
    ``intake_state`` fields; there is **no LLM** in that path.
    """

    model_config = ConfigDict(extra="forbid")

    # symptom booleans (keyword/phrase match on raw text)
    chest_pain: bool = False
    pain_radiation_arm_jaw_back: bool = False
    diaphoresis: bool = False
    dyspnea: bool = False
    nausea: bool = False
    face_droop: bool = False
    limb_weakness: bool = False
    speech_difficulty: bool = False
    sudden_vision_loss: bool = False
    sudden_confusion: bool = False
    worst_headache_ever: bool = False
    thunderclap_headache: bool = False
    neck_stiffness: bool = False
    fever: bool = False
    cant_breathe: bool = False
    throat_or_tongue_swelling: bool = False
    hives: bool = False
    known_allergen_exposure: bool = False
    suicidal_ideation: bool = False
    self_harm_intent: bool = False
    vaginal_bleeding: bool = False
    severe_abdominal_pain: bool = False
    vomiting_blood: bool = False
    rigid_abdomen: bool = False
    head_injury: bool = False
    on_anticoagulant: bool = False
    pregnant: bool = False

    # numerics (number parse; None if absent)
    sbp: int | None = None
    dbp: int | None = None
    glucose_mgdl: int | None = None
    spo2: int | None = None
    hr: int | None = None
    temp_f: int | None = None


# -------------------------------------------------------------------- intake state
class SlotValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    confidence: Confidence = Confidence.unknown
    source_msg_id: str | None = None
    updated_at: str | None = None


class IntakeState(BaseModel):
    """Per-session state, reloaded from SQLite each turn (stateless orchestration)."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    slots: dict[str, SlotValue] = Field(default_factory=dict)
    triage_floor: TriageBand = TriageBand.self_care
    floor_pinned: bool = False
    signals: Signals = Field(default_factory=Signals)
    status: str = "active"


class SafetyVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: EscalationLevel
    matched_rules: list[str] = Field(default_factory=list)
    source: EscalationSource
    rules_version: str = RULES_VERSION
    crisis: bool = False


# ---------------------------------------------------- SOAP (structured-output-safe)
class HPI(BaseModel):
    model_config = ConfigDict(extra="forbid")

    onset: str = ""
    location: str = ""
    duration: str = ""
    character: str = ""
    aggravating: str = ""
    relieving: str = ""
    timing: str = ""
    severity: str = ""


class PatientReportedVitals(BaseModel):
    """Closed vitals object (not a free-form dict).

    Native structured outputs require ``additionalProperties: false`` on every
    object, so the spec's ``patient_reported_vitals: dict`` is realised as a
    fixed-field model. Values are strings to allow patient phrasing ("186/122",
    "not measured").
    """

    model_config = ConfigDict(extra="forbid")

    sbp: str | None = None
    dbp: str | None = None
    glucose_mgdl: str | None = None
    spo2: str | None = None
    hr: str | None = None
    temp_f: str | None = None


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = ""
    url: str = ""
    chunk_id: str = ""


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = ""
    citation: Citation | None = None


class Subjective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chief_complaint: str = ""
    hpi: HPI = Field(default_factory=HPI)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    past_history: list[str] = Field(default_factory=list)
    social: str = ""
    low_confidence_fields: list[str] = Field(default_factory=list)


class Objective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_reported_vitals: PatientReportedVitals = Field(default_factory=PatientReportedVitals)
    notes: str = ""


class Triage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    band: TriageBand = TriageBand.self_care
    rationale: str = ""
    citations: list[Citation] = Field(default_factory=list)


class SOAP(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subjective: Subjective = Field(default_factory=Subjective)
    objective: Objective = Field(default_factory=Objective)
    observations: list[Observation] = Field(default_factory=list)
    triage: Triage = Field(default_factory=Triage)
    # Rule ids (spec section 12). The frontend derives its display count from
    # len(red_flags_checked); the list must hold the real rule ids (20 in Split 02),
    # never a hardcoded count. Kept as list[str] to stay structured-output-safe.
    red_flags_checked: list[str] = Field(default_factory=list)
    red_flags_triggered: list[str] = Field(default_factory=list)
    generated_at: str = ""
    disclaimer: str = DISCLAIMER


# ------------------------------------------------------------------------ tool I/O
class SlotUpdate(BaseModel):
    """One ``record_intake`` update (spec section 8)."""

    model_config = ConfigDict(extra="forbid")

    slot: str
    value: str
    confidence: Confidence = Confidence.unknown


class RecordIntakeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    updates: list[SlotUpdate] = Field(default_factory=list)


class RecordIntakeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    open_slots: list[str] = Field(default_factory=list)
    branch_hints: list[str] = Field(default_factory=list)


class RetrieveGuidelineInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    k: int = 5


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    source: str
    url: str
    score: float


class AssessEscalationInput(BaseModel):
    """Agent's independent escalation net (spec section 8). May escalate only."""

    model_config = ConfigDict(extra="forbid")

    level: EscalationLevel
    rationale: str


# ------------------------------------------------------------------ observability
class ToolCallTrace(BaseModel):
    """One row of the ``tool_calls`` audit/observability log (spec section 13)."""

    # protected_namespaces=() lets us keep the spec field name ``model``.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    session_id: str
    turn: int = 0
    tool: str
    args_json: str | None = None
    result_json: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    latency_ms: int | None = None
    model: str | None = None
    cost_usd: float = 0.0
    prompt_version: str = PROMPT_VERSION
    rules_version: str = RULES_VERSION
    ts: str | None = None

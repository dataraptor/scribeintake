"""Run-record + leaderboard schemas (Split 07).

These Pydantic models are what the harness emits, the runner persists to
``eval/runs/<ts>/*.jsonl``, and the metrics module consumes. Persisting **raw run records**
(not just aggregates) is deliberate: trend plots (Split 09/12) and the κ calibration work
(Split 08) need per-run data, and a discarded run can't be reconstructed.

The :class:`Leaderboard` is a **contract** the frontend Proof tab (Split 11/12) consumes; its
``ldDet`` / ``ldDist`` groups mirror the mockup's two-group shape (DETERMINISTIC · GATED vs
DISTRIBUTIONAL · TRACKED, NOT GATED) exactly, so keep those keys stable.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from scribeintake.models import EscalationLevel, EscalationSource, TriageBand

from .gate_check import LEVEL_RANK
from .scenario import ScenarioCategory

__all__ = [
    "TurnRecord",
    "ScenarioRun",
    "MetricValue",
    "LeaderboardMeta",
    "Leaderboard",
]


class TurnRecord(BaseModel):
    """Everything one patient turn produced inside a scenario run (the audit unit)."""

    model_config = ConfigDict(extra="forbid")

    turn: int
    user_msg: str
    escalation: EscalationLevel
    escalation_source: EscalationSource
    matched_rules: list[str] = Field(default_factory=list)
    crisis: bool = False
    triage_floor: TriageBand
    status: str
    assistant_text: str = ""
    tools_used: list[str] = Field(default_factory=list)
    # The safety_events written this turn ([{level, source}, ...]) — the gate logs one per turn
    # (CLEAR included), plus the agent's net when it escalates.
    safety_events: list[dict] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    model: str | None = None

    @property
    def escalated(self) -> bool:
        return self.escalation is not EscalationLevel.CLEAR


class ScenarioRun(BaseModel):
    """One end-to-end run of one scenario (×N runs share a ``scenario_id``, differ by label)."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    category: ScenarioCategory
    heldout: bool = False
    # Distinguishes the N repeats of the same scenario (``run-1`` ... ``run-N``).
    seed_label: str
    turns: list[TurnRecord] = Field(default_factory=list)
    final_soap: dict | None = None
    predicted_band: TriageBand | None = None
    intake_halted: bool = False
    final_status: str = "active"
    n_turns_run: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def max_level(self) -> EscalationLevel:
        """The peak escalation level across the run's turns."""
        best = EscalationLevel.CLEAR
        for t in self.turns:
            if LEVEL_RANK[t.escalation] > LEVEL_RANK[best]:
                best = t.escalation
        return best

    @property
    def escalated(self) -> bool:
        return self.max_level is not EscalationLevel.CLEAR


class MetricValue(BaseModel):
    """One metric on the leaderboard, honestly tagged ``reproducible`` (gated) or not (tracked).

    ``value`` is the numeric result (a fraction in ``[0,1]`` for rates, a count for
    "frozen must-escalate misses", or ``None`` when pending/not-applicable). ``spread`` is the
    stdev across the N aggregation rounds for distributional metrics. ``display`` is the
    pre-formatted cell the leaderboard renders.
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    group: str  # "deterministic" | "distributional"
    reproducible: bool
    value: float | None = None
    spread: float | None = None
    display: str
    # Only meaningful for ``group == "deterministic"``: whether the gated target is met.
    passing: bool = True
    pending: bool = False
    # Sample size behind the number (rounds for distributional, cases for deterministic).
    n: int | None = None
    note: str | None = None


class LeaderboardMeta(BaseModel):
    """Run metadata + the cost/latency summary (spec section 18)."""

    model_config = ConfigDict(extra="forbid")

    generated_at: str
    n_runs: int  # the requested ×N
    rounds: int  # rounds actually aggregated for distributional metrics (0 if deterministic-only)
    scenario_count: int
    deterministic_only: bool
    models: dict[str, str] = Field(default_factory=dict)
    mean_cost_usd: float | None = None
    mean_tokens: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None


class Leaderboard(BaseModel):
    """The published, two-group leaderboard (gated vs tracked) — JSON + Markdown artifact."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    meta: LeaderboardMeta
    framing: str
    metrics: list[MetricValue] = Field(default_factory=list)
    # Frontend Proof-tab shape (mockup ``ldDet`` / ``ldDist``). Serialized with the camelCase
    # keys the mockup uses so Split 11/12 can bind without a transform.
    ld_det: list[dict] = Field(default_factory=list, serialization_alias="ldDet", alias="ldDet")
    ld_dist: list[dict] = Field(default_factory=list, serialization_alias="ldDist", alias="ldDist")

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
    "Verdict",
    "AggregateVerdict",
    "KappaReport",
    "RetrievalReport",
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
    # Split 08: judge-calibration κ (one row per metric + ``overall``) and the RAGAS retrieval
    # summary. Empty/None on the deterministic-only tier (these are LLM/index-dependent).
    kappa: list[dict] = Field(default_factory=list)
    retrieval: dict | None = None


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


# ============================================================ judge verdicts (Split 08)
class Verdict(BaseModel):
    """One LLM-judge verdict for one metric on one item (the structured-output target).

    The judge model is forced to emit exactly this shape, so the parse is guaranteed. ``metric``
    and ``abstained`` are stamped in code after parsing (the model fills ``passed``/``score``/
    ``reason``/``span``): ``abstained`` records a refused/errored judge call so it is counted
    neither as a pass nor a fail (spec section 18 / section 21 #7).

    Kept structured-output-safe (plain types, no length/range constraints) like the SOAP schema.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str = ""
    passed: bool = False
    score: float = 0.0  # fraction in [0,1]; for boolean metrics, 1.0 if passed else 0.0
    reason: str = ""
    span: str = ""  # the offending text span ("" when nothing offends)
    abstained: bool = False


class AggregateVerdict(BaseModel):
    """The N-run majority for one metric on one item (the honesty unit, section 15).

    A single judge pass is non-deterministic, so the reported verdict is the **majority** of N
    passes, with ``agreement`` = the fraction of (non-abstained) passes that match it. A judge
    that flip-flops surfaces as low agreement instead of a falsely-confident single number.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str
    passed: bool  # the majority verdict over the non-abstained runs
    mean: float  # mean score over the non-abstained runs
    agreement: float  # fraction of non-abstained runs agreeing with ``passed``
    n: int  # total judge runs requested
    n_effective: int  # non-abstained runs (the ones the verdict is computed from)
    verdicts: list[Verdict] = Field(default_factory=list)

    @property
    def all_abstained(self) -> bool:
        return self.n_effective == 0


class KappaReport(BaseModel):
    """Cohen's κ between the judge majority and the human label for one metric (section 15).

    ``kappa`` is ``None`` when undefined (a single-class label set — everything agrees by
    default, so κ is meaningless and must not be reported as 0). The confusion counts and
    ``interpretation`` (κ ≥ 0.6 substantial / ≥ 0.8 strong) are reported alongside.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str
    kappa: float | None
    n: int
    both_pass: int = 0
    both_fail: int = 0
    judge_pass_human_fail: int = 0
    judge_fail_human_pass: int = 0
    observed_agreement: float = 0.0
    expected_agreement: float = 0.0
    interpretation: str = ""
    note: str | None = None


class RetrievalReport(BaseModel):
    """RAGAS-style retrieval metrics over a held-out query→relevant-chunk label set (section 11).

    A **transparent local implementation** (not the LLM-judged RAGAS package — recorded in
    ``impl``): context precision/recall are computed from the labels; faithfulness/answer
    relevancy are content-term-overlap proxies. Every number is unit-testable on a fixture.
    """

    model_config = ConfigDict(extra="forbid")

    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    n_queries: int
    k: int
    impl: str = "local-transparent-v1"
    per_query: list[dict] = Field(default_factory=list)

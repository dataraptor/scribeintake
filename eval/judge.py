"""LLM-as-judge + Cohen's κ calibration (Split 08, spec section 15).

The distributional **quality** layer — and the meta-eval that proves the judge itself is
trustworthy. Three rubric-driven judges grade the product's outputs:

* **faithfulness** — every health claim in a SOAP is backed by a cited ``chunk_id`` whose text
  *supports* it (an uncited or present-but-irrelevant citation fails; section 21 #4);
* **no-diagnosis** — the assistant never diagnosed / named a disease / prescribed;
* **no-coaching-after-escalation** — after an EMERGENCY, the reply let the template stand.

A single judge pass is non-deterministic, so trust comes from three things, **not** from any
sampling knob (this provider, like Opus, rejects them — the client never sends any):

1. a **tight structured rubric** per metric (versioned markdown in ``scribeintake/prompts/``),
2. **N-run majority** with reported agreement (:func:`judge_majority`), and
3. a **human-calibration subset** with reported **Cohen's κ** (:func:`run_calibration`) — if κ
   is low, the rubric is the deliverable to fix, never the human labels.

Provider note: the spec pins Opus 4.8 (``config.MODEL_JUDGE``); this environment ships an Azure
GPT-5.5 key (the Split-03 deviation), so the judge runs on the same structured-output deployment
as the terminal summary call — guaranteed-parse verdicts via ``StructuredClient.parse`` with
``reasoning_effort`` high. Judge-backed numbers are **reported, never CI-gated** (section 15).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from statistics import mean, stdev

from pydantic import BaseModel, ConfigDict

from scribeintake.config import MAX_TRIAGE_TOKENS, settings
from scribeintake.llm import STOP_MAX_TOKENS, STOP_REFUSAL, StructuredClient

from .metrics import DISTRIBUTIONAL, _fmt_mean_spread, _rounds
from .models import AggregateVerdict, KappaReport, MetricValue, ScenarioRun, Verdict
from .scenario import Scenario

# Judge reasoning route: the SOAP/triage-grade clinical-reasoning effort (a tool-less structured
# call, so the route is accepted). Bounded output — a verdict is small.
JUDGE_EFFORT = "high"
JUDGE_MAX_TOKENS = MAX_TRIAGE_TOKENS

FAITHFULNESS = "faithfulness"
NO_DIAGNOSIS = "no_diagnosis"
NO_COACHING = "no_coaching_after_escalation"  # the leaderboard metric key

# Calibration-case metric vocabulary (the dispatch keys in :func:`judge_calibration_case`).
# ``no_coaching`` is the short form of the leaderboard's ``no_coaching_after_escalation``.
CAL_NO_COACHING = "no_coaching"
CALIBRATION_METRICS = frozenset({FAITHFULNESS, NO_DIAGNOSIS, CAL_NO_COACHING})


# ============================================================ rubric loading
def load_rubric(name: str) -> str:
    """Read a versioned judge rubric markdown shipped in ``scribeintake/prompts/``."""
    return resources.files("scribeintake").joinpath(f"prompts/{name}.md").read_text(
        encoding="utf-8"
    )


# ============================================================ the judge client
def build_judge_client() -> StructuredClient:
    """Build the live structured-output judge client (same Azure deployment as the summary call).

    Kept as a named seam so a dedicated Opus judge deployment can be swapped in later without
    touching callers. Raises a clear error only when invoked without credentials.
    """
    from scribeintake.llm import build_summary_client

    return build_summary_client(settings)


def _abstain(metric: str, reason: str) -> Verdict:
    """A judge call that refused / errored: counted as neither pass nor fail (section 21 #7)."""
    return Verdict(metric=metric, passed=False, score=0.0, reason=reason, abstained=True)


def _judge(
    client: StructuredClient,
    *,
    rubric: str,
    content: str,
    metric: str,
    effort: str = JUDGE_EFFORT,
    max_tokens: int = JUDGE_MAX_TOKENS,
) -> Verdict:
    """One structured-output judge call → a stamped :class:`Verdict` (abstains on refusal/error)."""
    try:
        resp = client.parse(
            system=rubric,
            messages=[{"role": "user", "content": content}],
            schema=Verdict,
            effort=effort,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - a judge crash must not crash the eval
        return _abstain(metric, f"judge error: {type(exc).__name__}")

    # Section 18: check stop_reason before trusting content; a refused/truncated call abstains.
    if resp.refused or resp.parsed is None or resp.stop_reason in (STOP_REFUSAL, STOP_MAX_TOKENS):
        return _abstain(metric, f"judge {resp.stop_reason}")

    verdict: Verdict = resp.parsed  # type: ignore[assignment]
    # Stamp the metric (don't trust the model's free-text ``metric``) and align score for the
    # boolean metrics so the leaderboard mean and κ agree on a single source of truth.
    score = verdict.score if metric == FAITHFULNESS else (1.0 if verdict.passed else 0.0)
    return verdict.model_copy(
        update={"metric": metric, "abstained": False, "score": _clamp01(score)}
    )


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


# ============================================================ the three judges
def judge_faithfulness(soap: dict, chunks: dict[str, str], *, client: StructuredClient) -> Verdict:
    """Judge whether every health claim in ``soap`` is grounded in its cited chunk text.

    ``chunks`` maps ``chunk_id`` → chunk text. Observations citing a real chunk whose text
    supports the claim pass; uncited or present-but-irrelevant citations fail.
    """
    content = _faithfulness_content(soap, chunks)
    rubric = load_rubric("judge_faithfulness")
    return _judge(client, rubric=rubric, content=content, metric=FAITHFULNESS)


def judge_no_diagnosis(transcript: str | list[dict], *, client: StructuredClient) -> Verdict:
    """Judge whether the assistant transcript stayed within intake scope (no diagnosis/Rx/dose)."""
    content = "ASSISTANT TRANSCRIPT:\n\n" + _transcript_text(transcript)
    rubric = load_rubric("judge_no_diagnosis")
    return _judge(client, rubric=rubric, content=content, metric=NO_DIAGNOSIS)


def judge_no_coaching(reply: str | dict, *, client: StructuredClient) -> Verdict:
    """Judge whether the post-EMERGENCY reply let the emergency template stand (no coaching)."""
    content = "ASSISTANT REPLY ON THE EMERGENCY TURN:\n\n" + _reply_text(reply)
    rubric = load_rubric("judge_no_coaching")
    return _judge(client, rubric=rubric, content=content, metric=NO_COACHING)


# --------------------------------------------------------- content builders
def _faithfulness_content(soap: dict, chunks: dict[str, str]) -> str:
    observations = soap.get("observations") or []
    lines = ["OBSERVATIONS (health claims to audit):"]
    cited_ids: list[str] = []
    if not observations:
        lines.append("  (none)")
    for i, obs in enumerate(observations, start=1):
        text = (obs or {}).get("text", "")
        citation = (obs or {}).get("citation")
        if citation and citation.get("chunk_id"):
            cid = citation["chunk_id"]
            cited_ids.append(cid)
            lines.append(f"  {i}. claim: {text}\n     cites: {cid}")
        else:
            lines.append(f"  {i}. claim: {text}\n     cites: uncited")

    triage = soap.get("triage") or {}
    for c in triage.get("citations") or []:
        if c.get("chunk_id"):
            cited_ids.append(c["chunk_id"])

    lines.append("\nCITED CHUNK TEXT (chunk_id -> guideline text):")
    seen: set[str] = set()
    any_chunk = False
    for cid in cited_ids:
        if cid in seen:
            continue
        seen.add(cid)
        any_chunk = True
        text = chunks.get(cid, "TEXT NOT FOUND for this chunk_id")
        lines.append(f"  [{cid}] {text}")
    if not any_chunk:
        lines.append("  (no chunks were cited)")
    return "\n".join(lines)


def _transcript_text(transcript: str | list[dict]) -> str:
    if isinstance(transcript, str):
        return transcript.strip() or "(empty)"
    parts: list[str] = []
    for msg in transcript:
        role = (msg or {}).get("role", "assistant")
        if role != "assistant":
            continue
        text = (msg or {}).get("text") or (msg or {}).get("content") or ""
        if text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts) or "(empty)"


def _reply_text(reply: str | dict) -> str:
    if isinstance(reply, str):
        return reply.strip() or "(empty)"
    text = (reply or {}).get("assistant_text") or (reply or {}).get("text") or ""
    return text.strip() or "(empty)"


# ============================================================ N-run majority
def judge_majority(judge_fn: Callable[[], Verdict], *, n: int = 3) -> AggregateVerdict:
    """Run ``judge_fn`` ``n`` times and report the **majority** verdict + agreement.

    Abstained verdicts (refusal/error) are excluded from the vote. The decision is a strict
    majority of the non-abstained passes (a tie resolves to *fail* — conservative for these
    "did the product behave well?" metrics). ``agreement`` is the fraction of non-abstained
    verdicts on the majority side; a flip-flopping judge surfaces as low agreement.
    """
    verdicts = [judge_fn() for _ in range(max(1, n))]
    metric = next((v.metric for v in verdicts if v.metric), "")
    effective = [v for v in verdicts if not v.abstained]
    if not effective:
        return AggregateVerdict(
            metric=metric, passed=False, mean=0.0, agreement=0.0, n=len(verdicts),
            n_effective=0, verdicts=verdicts,
        )
    n_pass = sum(1 for v in effective if v.passed)
    n_fail = len(effective) - n_pass
    passed = n_pass > n_fail  # strict majority; tie -> fail
    agreement = max(n_pass, n_fail) / len(effective)
    return AggregateVerdict(
        metric=metric,
        passed=passed,
        mean=mean(v.score for v in effective),
        agreement=agreement,
        n=len(verdicts),
        n_effective=len(effective),
        verdicts=verdicts,
    )


# ============================================================ Cohen's κ
def cohens_kappa(pairs: list[tuple[bool, bool]], *, metric: str = "overall") -> KappaReport:
    """Cohen's κ between two binary raters over ``pairs`` of ``(judge_passed, human_passed)``.

    Returns a :class:`KappaReport` with the 2×2 confusion counts, observed/expected agreement,
    κ, and an interpretation. κ is **undefined (``None``)** when the labels are single-class
    (expected agreement 1.0 — everything agrees by default, so κ is meaningless and must not be
    reported as 0); the report says so explicitly.
    """
    n = len(pairs)
    both_pass = sum(1 for j, h in pairs if j and h)
    both_fail = sum(1 for j, h in pairs if not j and not h)
    judge_pass_human_fail = sum(1 for j, h in pairs if j and not h)
    judge_fail_human_pass = sum(1 for j, h in pairs if not j and h)

    if n == 0:
        return KappaReport(
            metric=metric, kappa=None, n=0, interpretation="undefined (no cases)",
            note="no calibration cases for this metric",
        )

    p_o = (both_pass + both_fail) / n
    judge_pass = (both_pass + judge_pass_human_fail) / n
    human_pass = (both_pass + judge_fail_human_pass) / n
    p_e = judge_pass * human_pass + (1 - judge_pass) * (1 - human_pass)

    single_class = (1 - p_e) < 1e-12
    kappa = None if single_class else (p_o - p_e) / (1 - p_e)
    return KappaReport(
        metric=metric,
        kappa=kappa,
        n=n,
        both_pass=both_pass,
        both_fail=both_fail,
        judge_pass_human_fail=judge_pass_human_fail,
        judge_fail_human_pass=judge_fail_human_pass,
        observed_agreement=p_o,
        expected_agreement=p_e,
        interpretation=_interpret_kappa(kappa, single_class),
        note=("single-class labels — κ undefined; add both-class cases" if single_class else None),
    )


def _interpret_kappa(kappa: float | None, single_class: bool) -> str:
    if single_class or kappa is None:
        return "undefined (single-class labels)"
    if kappa < 0:
        return "poor (worse than chance)"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "strong (near-perfect)"


# ============================================================ calibration subset
class CalibrationCase(BaseModel):
    """One hand-labelled calibration case (``eval/calibration/cases.yaml``).

    Exactly one metric per case with the **human** verdict (``human_passed``) and a rationale,
    plus the metric-specific payload (SOAP+chunks for faithfulness, a transcript for
    no-diagnosis, a reply for no-coaching). κ is meaningless if all labels are one class, so the
    set deliberately covers **both** classes per metric.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    metric: str
    human_passed: bool
    rationale: str
    soap: dict | None = None
    chunks: list[dict] | None = None  # [{chunk_id, text}]
    transcript: list[dict] | str | None = None
    reply: str | None = None


DEFAULT_CALIBRATION = Path(__file__).resolve().parent / "calibration" / "cases.yaml"


def load_calibration_cases(path: str | Path | None = None) -> list[CalibrationCase]:
    """Load + validate the calibration cases (defaults to ``eval/calibration/cases.yaml``)."""
    import yaml

    p = Path(path) if path else DEFAULT_CALIBRATION
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{p}: calibration file must be a YAML list of cases")
    cases = [CalibrationCase.model_validate(c) for c in raw]
    for c in cases:
        if c.metric not in CALIBRATION_METRICS:
            raise ValueError(f"{p}: case {c.id!r} has unknown metric {c.metric!r}")
    return cases


def judge_calibration_case(
    case: CalibrationCase, *, client: StructuredClient, n: int = 3
) -> AggregateVerdict:
    """Run the metric-appropriate judge ``n`` times over one calibration case → majority."""
    if case.metric == FAITHFULNESS:
        chunks = {c["chunk_id"]: c["text"] for c in (case.chunks or [])}
        soap = case.soap or {}
        return judge_majority(lambda: judge_faithfulness(soap, chunks, client=client), n=n)
    if case.metric == NO_DIAGNOSIS:
        transcript = case.transcript or ""
        return judge_majority(lambda: judge_no_diagnosis(transcript, client=client), n=n)
    reply = case.reply or ""
    return judge_majority(lambda: judge_no_coaching(reply, client=client), n=n)


def run_calibration(
    cases: list[CalibrationCase], *, client: StructuredClient, n: int = 3
) -> list[KappaReport]:
    """Judge every calibration case (majority of ``n``) and compute κ per metric + overall.

    Returns one :class:`KappaReport` per metric present, followed by a pooled ``overall`` report.
    Abstained majorities are skipped from κ (they are neither a pass nor a fail).
    """
    pairs_by_metric: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    all_pairs: list[tuple[bool, bool]] = []
    for case in cases:
        agg = judge_calibration_case(case, client=client, n=n)
        if agg.all_abstained:
            continue
        pair = (agg.passed, case.human_passed)
        pairs_by_metric[case.metric].append(pair)
        all_pairs.append(pair)

    reports = [cohens_kappa(pairs, metric=m) for m, pairs in sorted(pairs_by_metric.items())]
    reports.append(cohens_kappa(all_pairs, metric="overall"))
    return reports


# ============================================================ leaderboard cells
def chunk_text_map(retriever: object | None) -> dict[str, str]:
    """Build a ``chunk_id`` → text map for faithfulness grounding.

    Uses the retriever's loaded records when present; otherwise rebuilds the (deterministic,
    model-free) corpus records so faithfulness can run even without a built index.
    """
    records = getattr(retriever, "records", None)
    if isinstance(records, dict) and records:
        return {cid: getattr(r, "text", "") for cid, r in records.items()}
    try:
        from scribeintake.rag.ingest import build_records

        return {r.chunk_id: r.text for r in build_records(settings.KB_DIR)}
    except Exception:  # noqa: BLE001 - no corpus available -> empty map (faithfulness abstains)
        return {}


def compute_judge_metrics(
    scenarios: list[Scenario],
    runs: list[ScenarioRun],
    *,
    client: StructuredClient,
    retriever: object | None = None,
) -> list[MetricValue]:
    """The three judge-backed distributional cells, mean ± stdev **over rounds** (section 15).

    Each run is judged once per round; the cross-round spread is the reported distribution
    (``judge_majority`` is reserved for the κ calibration, where a single fixed case needs
    internal repeats). Faithfulness needs a completed SOAP with observations (often absent on a
    short run → that round contributes nothing, the cell stays honest, never a fabricated 0).
    """
    chunks = chunk_text_map(retriever)
    rounds = _rounds(runs)

    faithfulness = _judge_dist(
        FAITHFULNESS, "Faithfulness",
        [_round_faithfulness(rnd, client, chunks) for rnd in rounds],
    )
    no_diagnosis = _judge_dist(
        NO_DIAGNOSIS, "No-diagnosis",
        [_round_no_diagnosis(rnd, client) for rnd in rounds],
    )
    no_coaching = _judge_dist(
        NO_COACHING, "No-coaching after escalation",
        [_round_no_coaching(rnd, client) for rnd in rounds],
    )
    return [faithfulness, no_diagnosis, no_coaching]


def _round_faithfulness(
    rnd: list[ScenarioRun], client: StructuredClient, chunks: dict[str, str]
) -> float | None:
    scores: list[float] = []
    for run in rnd:
        soap = run.final_soap
        if not soap or not (soap.get("observations")):
            continue
        v = judge_faithfulness(soap, chunks, client=client)
        if not v.abstained:
            scores.append(v.score)
    return mean(scores) if scores else None


def _round_no_diagnosis(rnd: list[ScenarioRun], client: StructuredClient) -> float | None:
    flags: list[float] = []
    for run in rnd:
        transcript = [
            {"role": "assistant", "text": t.assistant_text} for t in run.turns if t.assistant_text
        ]
        if not transcript:
            continue
        v = judge_no_diagnosis(transcript, client=client)
        if not v.abstained:
            flags.append(1.0 if v.passed else 0.0)
    return mean(flags) if flags else None


def _round_no_coaching(rnd: list[ScenarioRun], client: StructuredClient) -> float | None:
    from scribeintake.models import EscalationLevel

    flags: list[float] = []
    for run in rnd:
        emergency_turns = [t for t in run.turns if t.escalation is EscalationLevel.EMERGENCY]
        if not emergency_turns:
            continue
        reply = emergency_turns[0].assistant_text
        v = judge_no_coaching(reply, client=client)
        if not v.abstained:
            flags.append(1.0 if v.passed else 0.0)
    return mean(flags) if flags else None


def _judge_dist(key: str, label: str, round_values: list[float | None]) -> MetricValue:
    """Aggregate per-round judge values into a mean ± stdev leaderboard cell (honest pending)."""
    vals = [v for v in round_values if v is not None]
    if not vals:
        return MetricValue(
            key=key, label=label, group=DISTRIBUTIONAL, reproducible=False, value=None,
            display="pending", pending=True, n=0,
            note="no judgeable runs (need completed/escalated runs with a key)",
        )
    m = mean(vals)
    sd = stdev(vals) if len(vals) > 1 else 0.0
    return MetricValue(
        key=key, label=label, group=DISTRIBUTIONAL, reproducible=False, value=m, spread=sd,
        display=_fmt_mean_spread(m, sd), n=len(vals),
    )

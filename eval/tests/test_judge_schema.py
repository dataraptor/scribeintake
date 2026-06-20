"""Judge schema / majority / abstain — deterministic (no API key, no model).

The Opus/GPT call is replaced by a scripted :class:`ScriptedJudge` that returns queued
:class:`~eval.models.Verdict` objects (or refuses), so the judge *logic* — verdict stamping,
score alignment, N-run majority + agreement, and the refusal→abstain path — is covered without
the network. The judge's real correctness on text is the live tier's job (``test_judge_live``).
"""

from __future__ import annotations

import inspect

from eval import judge
from eval.judge import (
    FAITHFULNESS,
    NO_DIAGNOSIS,
    judge_faithfulness,
    judge_majority,
    judge_no_coaching,
    judge_no_diagnosis,
)
from eval.models import Verdict
from scribeintake.llm import STOP_END_TURN, STOP_REFUSAL, LLMUsage, StructuredResponse


class ScriptedJudge:
    """A :class:`StructuredClient` that returns queued verdicts; ``None`` queues a refusal."""

    model = "gpt-5.5"

    def __init__(self, verdicts: list[Verdict | None]) -> None:
        self._verdicts = list(verdicts)
        self.calls = 0

    def parse(self, *, system, messages, schema, effort="high", max_tokens=2048):
        self.calls += 1
        v = self._verdicts.pop(0) if self._verdicts else None
        if v is None:
            return StructuredResponse(
                parsed=None, refused=True, stop_reason=STOP_REFUSAL, usage=LLMUsage(),
                model=self.model,
            )
        return StructuredResponse(
            parsed=v, refused=False, stop_reason=STOP_END_TURN, usage=LLMUsage(), model=self.model,
        )


def _v(passed: bool, score: float, **kw) -> Verdict:
    return Verdict(passed=passed, score=score, reason="r", **kw)


SOAP = {"observations": [{"text": "claim", "citation": {"chunk_id": "chk_x"}}]}
CHUNKS = {"chk_x": "supporting text"}


# ------------------------------------------------------------------ verdict stamping
def test_verdict_parses_and_metric_is_stamped():
    client = ScriptedJudge([_v(True, 1.0)])
    v = judge_faithfulness(SOAP, CHUNKS, client=client)
    assert v.metric == FAITHFULNESS
    assert v.passed is True
    assert v.abstained is False
    assert v.score == 1.0
    assert client.calls == 1


def test_boolean_metric_score_is_aligned_to_passed():
    # A no-diagnosis verdict is boolean: a stray fractional ``score`` must snap to passed↔1/0.
    client = ScriptedJudge([_v(False, 0.7)])
    v = judge_no_diagnosis("the assistant text", client=client)
    assert v.metric == NO_DIAGNOSIS
    assert v.passed is False
    assert v.score == 0.0

    client = ScriptedJudge([_v(True, 0.2)])
    assert judge_no_diagnosis("ok", client=client).score == 1.0


def test_faithfulness_score_is_preserved():
    # Faithfulness is a fraction, not a boolean — the model's score is kept (not snapped).
    client = ScriptedJudge([_v(False, 0.5)])
    v = judge_faithfulness(SOAP, CHUNKS, client=client)
    assert v.score == 0.5
    assert v.passed is False


# --------------------------------------------------------------------- abstain path
def test_refusal_yields_abstain_not_crash_or_false_pass():
    client = ScriptedJudge([None])  # the judge refuses
    v = judge_no_coaching("call 911 now", client=client)
    assert v.abstained is True
    assert v.passed is False  # an abstain is never a pass
    assert "refusal" in v.reason.lower() or "abstain" in v.reason.lower() or v.reason


# ------------------------------------------------------------------- N-run majority
def test_majority_takes_majority_and_reports_agreement():
    client = ScriptedJudge([_v(True, 1.0), _v(True, 1.0), _v(False, 0.0)])
    agg = judge_majority(lambda: judge_no_diagnosis("x", client=client), n=3)
    assert agg.passed is True  # 2 of 3
    assert abs(agg.agreement - 2 / 3) < 1e-9
    assert agg.n == 3 and agg.n_effective == 3
    assert abs(agg.mean - 2 / 3) < 1e-9


def test_majority_tie_resolves_to_fail():
    client = ScriptedJudge([_v(True, 1.0), _v(False, 0.0)])
    agg = judge_majority(lambda: judge_no_diagnosis("x", client=client), n=2)
    assert agg.passed is False  # 1-1 tie → conservative fail
    assert agg.agreement == 0.5


def test_majority_excludes_abstain_from_the_vote():
    client = ScriptedJudge([_v(True, 1.0), None, _v(True, 1.0)])
    agg = judge_majority(lambda: judge_no_diagnosis("x", client=client), n=3)
    assert agg.n == 3
    assert agg.n_effective == 2  # the abstain is dropped
    assert agg.passed is True
    assert agg.agreement == 1.0


def test_majority_all_abstain_is_flagged():
    client = ScriptedJudge([None, None, None])
    agg = judge_majority(lambda: judge_no_diagnosis("x", client=client), n=3)
    assert agg.all_abstained is True
    assert agg.n_effective == 0
    assert agg.passed is False
    assert agg.mean == 0.0


# -------------------------------------------------------------- §6 grep (no knobs)
def test_judge_source_sends_no_sampling_knobs():
    src = inspect.getsource(judge)
    for bad in ("temperature", "top_p", "top_k", '"seed"', "seed="):
        assert bad not in src, f"forbidden sampling knob in judge.py: {bad}"

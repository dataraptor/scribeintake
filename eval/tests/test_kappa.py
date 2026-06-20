"""Cohen's κ math + calibration grouping — deterministic (no API key).

κ is the senior signal (spec section 15): the project evaluates its own evaluator. These tests
pin the math against hand-computed confusion matrices (perfect / chance / total-disagreement /
single-class) and check that ``run_calibration`` groups κ per metric + overall.
"""

from __future__ import annotations

from eval import judge
from eval.judge import cohens_kappa, load_calibration_cases, run_calibration
from eval.models import AggregateVerdict

T, F = True, False


# ----------------------------------------------------------------- the κ math
def test_perfect_agreement_kappa_is_one():
    pairs = [(T, T), (T, T), (F, F), (F, F)]
    rep = cohens_kappa(pairs)
    assert rep.kappa == 1.0
    assert rep.observed_agreement == 1.0
    assert "strong" in rep.interpretation


def test_chance_agreement_kappa_near_zero():
    # judge independent of human: each cell of the 2×2 equally likely → p_o == p_e → κ = 0.
    pairs = [(T, T), (T, F), (F, T), (F, F)]
    rep = cohens_kappa(pairs)
    assert abs(rep.kappa) < 1e-9
    assert rep.both_pass == 1 and rep.both_fail == 1
    assert rep.judge_pass_human_fail == 1 and rep.judge_fail_human_pass == 1


def test_total_disagreement_is_negative():
    pairs = [(T, F), (F, T)]
    rep = cohens_kappa(pairs)
    assert rep.kappa == -1.0
    assert "poor" in rep.interpretation


def test_single_class_labels_make_kappa_undefined_not_zero():
    rep = cohens_kappa([(T, T), (T, T), (T, T)])
    assert rep.kappa is None  # everything agrees by default → κ meaningless, NOT 0.0
    assert "undefined" in rep.interpretation
    assert rep.note and "single-class" in rep.note


def test_empty_pairs_kappa_undefined():
    rep = cohens_kappa([])
    assert rep.kappa is None
    assert rep.n == 0


def test_substantial_band_interpretation():
    # 9/10 agree, evenly split classes → κ in the "substantial" band.
    pairs = [(T, T)] * 4 + [(F, F)] * 4 + [(T, F)] + [(F, T)]
    rep = cohens_kappa(pairs)
    assert rep.kappa is not None and 0.6 <= rep.kappa < 0.8
    assert rep.interpretation == "substantial"


# ----------------------------------------------------------------- calibration grouping
def test_run_calibration_groups_per_metric_and_overall(monkeypatch):
    """With a judge that perfectly matches the human labels, every per-metric κ is 1.0."""
    cases = load_calibration_cases()

    def _perfect(case, *, client, n=3):
        return AggregateVerdict(
            metric=case.metric, passed=case.human_passed, mean=1.0, agreement=1.0,
            n=n, n_effective=n,
        )

    monkeypatch.setattr(judge, "judge_calibration_case", _perfect)
    reports = run_calibration(cases, client=object(), n=3)
    by_metric = {r.metric: r for r in reports}

    assert {"faithfulness", "no_diagnosis", "no_coaching", "overall"} <= set(by_metric)
    for r in reports:
        assert r.kappa == 1.0  # perfect agreement, both classes present per metric


def test_run_calibration_overall_pools_all_cases(monkeypatch):
    cases = load_calibration_cases()
    monkeypatch.setattr(
        judge,
        "judge_calibration_case",
        lambda case, *, client, n=3: AggregateVerdict(
            metric=case.metric, passed=case.human_passed, mean=1.0, agreement=1.0,
            n=n, n_effective=n,
        ),
    )
    reports = run_calibration(cases, client=object(), n=1)
    overall = next(r for r in reports if r.metric == "overall")
    assert overall.n == len(cases)  # all 15 cases pooled


def test_run_calibration_skips_abstained_majorities(monkeypatch):
    cases = load_calibration_cases()

    def _all_abstain(case, *, client, n=3):
        return AggregateVerdict(
            metric=case.metric, passed=False, mean=0.0, agreement=0.0, n=n, n_effective=0
        )

    monkeypatch.setattr(judge, "judge_calibration_case", _all_abstain)
    reports = run_calibration(cases, client=object(), n=3)
    overall = next(r for r in reports if r.metric == "overall")
    assert overall.n == 0  # every case abstained → no κ pairs
    assert overall.kappa is None

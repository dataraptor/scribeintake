"""The red-team report generator + the committed artifact (Split 13 §3.4) — NO API key.

Asserts the deterministic report: every deterministic assertion passes, the case buckets are
classified correctly, the tracked cells are honestly *pending* (never fabricated), the report is
byte-reproducible, and the committed ``eval/redteam_report.json`` is in sync with a freshly
generated one (so the checked-in artifact can never drift from the code).
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.redteam import (
    REDTEAM_GENERATED_AT,
    build_report,
    classify_case,
    deterministic_assertions,
    render_markdown,
)
from eval.scenario import load_scenarios

ADVERSARIAL_DIR = Path(__file__).resolve().parents[1] / "scenarios" / "adversarial"
COMMITTED_JSON = Path(__file__).resolve().parents[1] / "redteam_report.json"
SCENARIOS = load_scenarios(ADVERSARIAL_DIR)


def test_all_deterministic_assertions_pass():
    for a in deterministic_assertions(SCENARIOS):
        assert a["passed"], f"deterministic assertion failed: {a['assertion']} — {a['detail']}"


def test_report_separates_passed_from_tracked():
    report = build_report(SCENARIOS)
    assert report["deterministic_passed"] is True
    assert report["source"] == "deterministic"
    # Tracked cells are honestly pending in the key-free report (no fabricated rates).
    for cell in report["tracked"].values():
        assert cell["status"] == "pending"
        assert cell["value"] is None


def test_case_counts_cover_every_scenario():
    report = build_report(SCENARIOS)
    assert report["case_counts"]["total"] == len(SCENARIOS)
    summed = sum(v for k, v in report["case_counts"].items() if k != "total")
    assert summed == len(SCENARIOS)  # every scenario lands in exactly one bucket


def test_classification_buckets():
    buckets = {classify_case(s) for s in SCENARIOS}
    assert {"injection", "oblique", "deescalation", "contradiction"} <= buckets


def test_report_is_byte_reproducible():
    a = json.dumps(build_report(SCENARIOS), indent=2)
    b = json.dumps(build_report(SCENARIOS), indent=2)
    assert a == b  # no Date.now/random in the artifact path


def test_committed_artifact_is_in_sync():
    """The checked-in deterministic report must match a freshly generated one."""
    fresh = build_report(SCENARIOS, generated_at=REDTEAM_GENERATED_AT)
    committed = json.loads(COMMITTED_JSON.read_text(encoding="utf-8"))
    assert committed == fresh, (
        "eval/redteam_report.json is stale — regenerate with `python -m eval.redteam`."
    )


def test_markdown_renders_the_headline_and_both_groups():
    md = render_markdown(build_report(SCENARIOS))
    assert "injection cannot disable the code gate" in md
    assert "PASSED — deterministic" in md
    assert "TRACKED — distributional" in md

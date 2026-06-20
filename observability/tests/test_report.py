"""Deterministic tests for the cost report + dashboard rendering (Split 09, §16)."""

import json

import pytest

from observability.dashboard import render_html, write_dashboard
from observability.report import (
    render_markdown,
    report_from_db,
    seed_synthetic_db,
    write_report,
)


@pytest.fixture
def report(conn):
    seed_synthetic_db(conn)
    return report_from_db(conn, generated_at="2026-06-20T00:00:00Z", source="synthetic-demo",
                          models={"intake": "gpt-5.5", "summary": "gpt-5.5"})


def test_report_headline_numbers_are_honest(report):
    assert report.source == "synthetic-demo"
    assert report.cost.local_tool_cost_usd == 0.0
    # caching demonstrably saved money on the synthetic warm turns
    assert report.savings.pct_saved > 0
    assert report.savings.with_cache_usd < report.savings.no_cache_usd
    assert 0 < report.cache_hit_rate < 1
    assert "cache" in report.trace_cost_label and "saved" in report.trace_cost_label


def test_report_tool_usage_counts(report):
    # the synthetic seed has 3 agent_step rows and 3 record_intake rows
    assert report.tool_usage["agent_step"] == 3
    assert report.tool_usage["record_intake"] == 3
    assert report.tool_usage["build_summary"] == 1


def test_report_latency_split_and_first_summary_annotated(report):
    lat = report.latency
    assert lat.intake_n == 3
    assert lat.summary_n >= 0
    # build_summary is the first terminal call → annotated as compile, excluded from percentiles
    assert lat.first_summary_ms is not None


def test_write_report_emits_json_and_md(report, tmp_path):
    json_path, md_path = write_report(report, tmp_path)
    assert json_path.exists() and md_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["source"] == "synthetic-demo"
    assert data["savings"]["pct_saved"] > 0
    md = md_path.read_text(encoding="utf-8")
    assert "Cost & Observability" in md
    assert "saved" in md


def test_markdown_renders_without_error(report):
    md = render_markdown(report)
    assert md.startswith("# ScribeIntake")
    assert "Latency percentiles" in md


def test_dashboard_html_renders(report, tmp_path):
    html = render_html(report)
    assert "<html" in html and "Cost &amp; Observability" in html
    out = write_dashboard(report, tmp_path / "dash.html")
    assert out.exists()
    assert "saved" in out.read_text(encoding="utf-8")


def test_report_with_empty_db_does_not_crash(conn):
    # No traces at all → a zero report, not an exception or a NaN.
    rep = report_from_db(conn, generated_at="2026-06-20T00:00:00Z", source="empty")
    assert rep.cost.total_cost_usd == 0.0
    assert rep.savings.pct_saved == 0.0
    assert rep.cache_hit_rate == 0.0
    # renders cleanly
    assert render_markdown(rep).startswith("# ScribeIntake")


def test_synthetic_seed_row_count(conn):
    sid = seed_synthetic_db(conn)
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM tool_calls WHERE session_id = ?", (sid,)
    ).fetchone()["n"]
    # 3 agent_step + 3 record_intake + 1 retrieve_guideline + build_summary + suggest_triage
    assert n == 9

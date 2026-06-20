"""Deterministic tests for cache-aware cost accounting over the trace (Split 09, §16)."""

import json

import pytest

from observability import cost
from observability.report import seed_synthetic_db
from observability.trace import iter_run_records, read_tool_calls
from scribeintake import db, pricing
from scribeintake.agent import AgentResult, AgentStep
from scribeintake.llm import LLMUsage
from scribeintake.orchestrator import run_turn


# ----------------------------------------------------------------- breakdown / savings
def test_session_cost_uses_all_buckets_and_zeroes_local_rows(conn):
    sid = seed_synthetic_db(conn)
    cb = cost.session_cost(conn, sid)

    # Recompute the expected model cost straight from the seeded buckets.
    rows = read_tool_calls(conn, sid)
    expected = sum(
        pricing.cost_usd(r.model, r.input_tokens, r.output_tokens,
                         r.cache_creation_tokens, r.cache_read_tokens)
        for r in rows if r.model in pricing.PRICES
    )
    assert cb.total_cost_usd == pytest.approx(expected)
    # Local RAG / record_intake rows are free.
    assert cb.local_tool_cost_usd == 0.0
    assert cb.per_tool_cost_usd.get("retrieve_guideline", 0.0) == 0.0
    # Caching actually saved money on this trace (warm turns have cache reads).
    assert cb.savings.cache_read_tokens > 0
    assert 0.0 < cb.savings.pct_saved < 1.0
    assert cb.savings.with_cache_usd < cb.savings.no_cache_usd


def test_breakdown_cache_read_share_matches_tokens(conn):
    sid = seed_synthetic_db(conn)
    cb = cost.session_cost(conn, sid)
    rows = read_tool_calls(conn, sid)
    prompt = sum(r.input_tokens + r.cache_read_tokens + r.cache_creation_tokens
                 for r in rows if r.model in pricing.PRICES)
    cr = sum(r.cache_read_tokens for r in rows if r.model in pricing.PRICES)
    assert cb.savings.cache_read_share == pytest.approx(cr / prompt)


# ------------------------------------------------------------------ fleet over runs/
def test_fleet_cost_over_run_records(tmp_path):
    runs = tmp_path / "runs" / "20260620T000000Z"
    runs.mkdir(parents=True)
    recs = [
        {"scenario_id": "a", "category": "routine", "seed_label": "run-1",
         "total_cost_usd": 0.010, "total_input_tokens": 4000, "total_output_tokens": 500},
        {"scenario_id": "b", "category": "benign", "seed_label": "run-1",
         "total_cost_usd": 0.020, "total_input_tokens": 6000, "total_output_tokens": 800},
    ]
    (runs / "a.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    fc = cost.fleet_cost(tmp_path / "runs")
    assert fc.n_sessions == 2
    assert fc.total_cost_usd == pytest.approx(0.030)
    assert fc.mean_cost_usd == pytest.approx(0.015)
    assert fc.mean_tokens == pytest.approx((4500 + 6800) / 2)


def test_fleet_cost_missing_dir_is_empty(tmp_path):
    fc = cost.fleet_cost(tmp_path / "does-not-exist")
    assert fc.n_sessions == 0
    assert list(iter_run_records(tmp_path / "does-not-exist")) == []


# ------------------------------------------------------------------ jsonl export
def test_export_session_jsonl_one_line_per_row(conn, tmp_path):
    sid = seed_synthetic_db(conn)
    out = tmp_path / "trace.jsonl"
    cost.export_session_jsonl(conn, sid, out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(read_tool_calls(conn, sid))
    first = json.loads(lines[0])
    assert {"tool", "model", "input_tokens", "cache_read_tokens", "cost_usd"} <= first.keys()


# ------------------------------ bucket-honesty: the orchestrator must price all 4 buckets
class _FakeAgent:
    """Returns a pre-built result whose model call carries cache-read tokens."""

    def run_turn(self, *, history, user_content, ctx, effort="medium"):
        return AgentResult(
            assistant_text="Could you tell me a little more about that?",
            steps=[
                AgentStep(
                    model="gpt-5.5",
                    text="...",
                    tool_calls=[],
                    usage=LLMUsage(input_tokens=240, output_tokens=30, cache_read_tokens=1800),
                    latency_ms=1500,
                )
            ],
            tool_executions=[],
        )


def test_orchestrator_persists_four_bucket_cost_not_input_only(conn):
    """A cache-read turn must be logged at the (cheaper) 4-bucket price, never input-only."""
    sid = db.create_session(conn)
    run_turn(sid, "I've had a mild sore throat for two days.", conn=conn, agent=_FakeAgent())

    row = conn.execute(
        "SELECT cost_usd, input_tokens, output_tokens, cache_read_tokens FROM tool_calls "
        "WHERE session_id = ? AND tool = 'agent_step'",
        (sid,),
    ).fetchone()
    four_bucket = pricing.cost_usd("gpt-5.5", 240, 30, 0, 1800)
    input_only = pricing.no_cache_cost_usd("gpt-5.5", 240, 30, 0, 1800)
    assert row["cache_read_tokens"] == 1800
    assert row["cost_usd"] == pytest.approx(four_bucket)
    assert row["cost_usd"] < input_only  # caching honestly reflected, not erased

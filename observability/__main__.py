"""``python -m observability`` — generate the cost report + dashboard over the trace.

Sources, in order of preference:

1. ``--db PATH`` (or the live ``data/scribeintake.db``) if it holds ``tool_calls`` rows →
   ``source = "live-db"`` (the real, cache-aware numbers).
2. otherwise a small, clearly-labelled **synthetic** trace (``source = "synthetic-demo"``) so the
   artifact + dashboard always render with no API key — the committed, byte-reproducible default.

Writes ``observability/cost_report.{json,md}`` and ``observability/dashboard.html``. Reads only;
no model call, no service. (Use ``observability.cache_check`` for the live caching proof.)
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from scribeintake import db
from scribeintake.config import settings

from .dashboard import write_dashboard
from .report import (
    DEFAULT_OUT_DIR,
    DEFAULT_RUNS_DIR,
    report_from_db,
    seed_synthetic_db,
    write_report,
)


def _has_traces(conn: sqlite3.Connection) -> bool:
    try:
        n = conn.execute("SELECT COUNT(*) AS n FROM tool_calls").fetchone()["n"]
        return int(n) > 0
    except sqlite3.Error:
        return False


def _models() -> dict[str, str]:
    return {"intake": settings.ACTIVE_INTAKE_MODEL, "summary": settings.ACTIVE_INTAKE_MODEL}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ScribeIntake cost/observability report")
    parser.add_argument("--db", default=None, help="SQLite DB to read traces from (default: live)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--runs-dir", default=str(DEFAULT_RUNS_DIR), help="eval runs/ for the fleet trend"
    )
    parser.add_argument("--ts", default=None, help="override generated_at (reproducible artifact)")
    parser.add_argument(
        "--synthetic", action="store_true", help="force the synthetic-demo trace (no live DB)"
    )
    args = parser.parse_args(argv)

    ts = args.ts or datetime.now(UTC).isoformat()
    runs_dir = args.runs_dir if Path(args.runs_dir).exists() else None

    db_path = args.db or (None if args.synthetic else str(settings.DB_PATH))
    use_live = not args.synthetic and db_path is not None and Path(db_path).exists()

    if use_live:
        conn = db.connect(db_path)
        if not _has_traces(conn):
            use_live = False
            conn.close()

    if use_live:
        report = report_from_db(
            conn, generated_at=ts, source="live-db", models=_models(), runs_dir=runs_dir
        )
        conn.close()
    else:
        # No real trace yet → a reproducible synthetic demo (key-free, clearly labelled).
        conn = db.connect(":memory:")
        seed_synthetic_db(conn)
        report = report_from_db(
            conn, generated_at=ts, source="synthetic-demo", models=_models(), runs_dir=runs_dir
        )
        conn.close()

    if report.source == "synthetic-demo":
        report.notes.append(
            "Synthetic demonstration of the cache-aware pipeline (key-free, byte-reproducible). On "
            "the wired Azure GPT-5.5 deployment, prompt caching is verified LIVE on the terminal "
            "build_summary call (cache_read 0→1280 tok on warm repeats — see observability/"
            "cache_check.py); the agent loop's prefix does not surface cache hits on this "
            "deployment. Run `python -m observability` over a live DB for real per-session numbers."
        )

    out_dir = Path(args.out_dir)
    json_path, md_path = write_report(report, out_dir)
    html_path = write_dashboard(report, out_dir / "dashboard.html")

    # ASCII-only console (Windows cp1252 stdout can't encode the report's middle-dot/✅; those
    # live in the utf-8 markdown). Mirrors the Split 08 console convention.
    print(f"source        : {report.source}")
    print(f"trace cost    : ${report.cost.total_cost_usd:.4f} "
          f"(cache {report.savings.pct_saved * 100:.0f}% saved)")
    print(f"cache savings : {report.savings.pct_saved * 100:.0f}% "
          f"(${report.savings.with_cache_usd:.4f} vs ${report.savings.no_cache_usd:.4f})")
    print(f"cache-hit rate: {report.cache_hit_rate * 100:.0f}% of prompt tokens")
    print(f"latency intake: p50 {report.latency.intake_p50_ms:.0f} / "
          f"p95 {report.latency.intake_p95_ms:.0f} ms")
    print(f"wrote         : {json_path.name}, {md_path.name}, {html_path.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

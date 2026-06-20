# Windows/PowerShell equivalents of the Makefile targets.
# Usage: .\tasks.ps1 <task>   e.g.  .\tasks.ps1 test
param(
    [Parameter(Position = 0)]
    [ValidateSet("install", "install-rag", "lint", "fmt", "test", "test-live", "ingest", "eval", "eval-ci", "cost-report", "cache-check")]
    [string]$Task = "test"
)

switch ($Task) {
    "install"     { pip install -e "./core[dev]" }
    # Adds the local RAG models (embedder + cross-encoder reranker -> torch) for live retrieval.
    "install-rag" { pip install -e "./core[dev,rag]" }
    "lint"        { ruff check }
    "fmt"         { ruff format }
    "test"        { python -m pytest core/tests -m "not live" }
    "test-live"   { python -m pytest core/tests -m live }
    # Build the local guideline index (downloads the embedder on first run).
    "ingest"      { python -m scribeintake.rag.ingest }
    # Full distributional eval (x3) - needs LLM credentials; writes eval/leaderboard.{json,md}.
    "eval"        { python -m eval.run --n 3 }
    # Deterministic eval gate (the CI tier) - no API key; fails on any gated-metric regression.
    "eval-ci"     { python -m eval.run --deterministic-only }
    # Cost & observability report (Split 09) - reads live DB / eval runs/ (or synthetic demo).
    "cost-report" { python -m observability }
    # Live prompt-cache verification (needs LLM credentials): cold-vs-warm cache_read proof.
    "cache-check" { python -c "from observability.cache_check import run_cache_check, format_result; print(format_result(run_cache_check()))" }
}

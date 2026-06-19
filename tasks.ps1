# Windows/PowerShell equivalents of the Makefile targets.
# Usage: .\tasks.ps1 <task>   e.g.  .\tasks.ps1 test
param(
    [Parameter(Position = 0)]
    [ValidateSet("install", "install-rag", "lint", "fmt", "test", "test-live", "ingest")]
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
}

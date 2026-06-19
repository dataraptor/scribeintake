# POSIX task runner. Windows/PowerShell equivalents live in tasks.ps1.
# Canonical commands use `python -m ...` so they work without `make`.

.PHONY: install install-rag lint fmt test test-live ingest

install:
	pip install -e "./core[dev]"

# Adds the local RAG models (embedder + cross-encoder reranker → torch) for live retrieval.
install-rag:
	pip install -e "./core[dev,rag]"

lint:
	ruff check

fmt:
	ruff format

test:
	python -m pytest core/tests -m "not live"

test-live:
	python -m pytest core/tests -m live

# Build the local guideline index (downloads the embedder on first run).
ingest:
	python -m scribeintake.rag.ingest

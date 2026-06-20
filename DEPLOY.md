# Deploy & HIPAA-boundary posture

> **Educational demo on synthetic data only — not a diagnostic, triage, or crisis service.**
> No real PHI anywhere. Adult patients, US-English (988/911). See [`docs/compliance.md`](docs/compliance.md).

This document covers the **single-instance run** (the demo) and the **production posture** — the
property that makes ScribeIntake a drop-in for a HIPAA boundary: *everything is local except the
LLM call.*

---

## 1. Single-instance run (the demo)

The whole stack — the FastAPI service **and** the static UI — runs from one process. The API mounts
`app/` at `/` and serves the committed Proof artifacts at `/proof/*` (Split 11).

```bash
# 1. install core + the API web deps (api/ is import-only, resolved via the repo-root pythonpath)
pip install -e "./core[dev]" "fastapi>=0.110,<1" "uvicorn>=0.27,<1" "httpx>=0.27,<1"
#    or:  make install-api        (Windows:  .\tasks.ps1 install-api)

# 2. (optional, for live turns) put LLM credentials in .env — see §2 below.
#    The deterministic safety gate runs with NO key; emergencies/injections short-circuit key-free.

# 3. boot the one process and open http://localhost:8000
make demo                          # Windows:  .\tasks.ps1 demo
#    == python -m uvicorn api.main:app --port 8000
```

Endpoints: `GET /health`, `POST /session`, `POST /session/{id}/message` (SSE), `GET /session/{id}/summary`,
`GET /session/{id}/trace`, plus the static UI at `/` and `/proof/{leaderboard,cost_report}.json`.

### Optional containerization
A `Dockerfile` is intentionally **not** shipped for v1 (the one-command uvicorn run is the demo
surface). To containerize: base `python:3.11-slim`, `pip install -e "./core[dev]" fastapi uvicorn`,
copy the repo, `CMD ["python","-m","uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]`,
and mount `.env` / `data/` as a secret + volume (never bake them into the image).

---

## 2. Configuration & secrets

All secrets are env-only (`.env` is gitignored and **never** committed — enforced by
`scripts/audit.py`, which scans the tree *and* git history on every push).

| Env var | Purpose | Required for |
|---|---|---|
| `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT` | the wired GPT-5.5 deployment (Split 03) | live turns |
| `CHAT_LLM_MODEL` | model id (default `gpt-5.5`) | live turns |
| `ANTHROPIC_API_KEY` | the spec's Claude pin (forward-compat) | live turns (Claude path) |
| `API_CORS_ORIGINS` | CORS allowlist (default `*`) | split-origin UI |
| `DATA_DIR` | SQLite + RAG index location (default `<repo>/data`, gitignored) | persistence |

The deterministic tier, the per-commit gate, and the emergency/injection safety flows need **no
secret at all** — the gate is code.

---

## 3. The HIPAA-boundary posture (the healthcare differentiator)

The clinical-safety thesis is also a **deployment** thesis. Every component that touches patient
text runs **locally**; the **only** egress is the LLM call:

| Component | Location | Touches patient text? |
|---|---|---|
| Safety extractor + rule engine (`safety/`) | **local** (pure code) | yes — never leaves the box |
| Intake state machine (`intake/`) | **local** | yes |
| Embeddings + cross-encoder rerank + BM25 (`rag/`) | **local** (sentence-transformers, no API) | yes |
| SQLite storage (`db.py`) | **local** file | yes |
| Triage floor / clamp / summary assembly | **local** code | yes |
| **LLM call** (intake loop, SOAP/triage) | **remote** | **yes — the one boundary** |

`scripts/audit.py` asserts this (`local_only`): the safety + RAG paths contain no `anthropic` /
`openai` / `requests.post` / `httpx.post` token — they cannot call out.

**To run inside a HIPAA boundary you change one thing:** point the LLM client at a **BAA-covered
endpoint** — Amazon Bedrock, Google Vertex, or the Anthropic/Azure platform under a signed BAA.
Nothing else moves: embeddings, rerank, BM25, the safety gate, and storage are already local, so no
PHI is sent to any third party other than that one BAA-covered model endpoint. The `StructuredClient`
/ agent-loop client seam (Split 03) is the single place to re-point.

### What is **not** production-hardened in this demo
- Single-instance, single-process; no auth/RBAC, no rate limiting, no TLS termination (front it with
  a reverse proxy + a BAA-covered model endpoint for real use).
- SQLite (fine for the demo; swap for a managed encrypted store behind the same `db.py` seam).
- Synthetic data only — **no real PHI** is in scope for v1.

---

## 4. Pre-deploy checklist

Run the acceptance gate and read the sign-off matrix:

```bash
make acceptance-ci          # no key — gated rows (suite, eval, invariants, audit, safety e2e)
make acceptance             # with a key in .env — adds the live model row + measured perf
cat acceptance_report.md    # the sign-off matrix
```

See [`docs/release-checklist.md`](docs/release-checklist.md) for the full human-runnable checklist.

# Changelog

All notable changes to ScribeIntake.

## [1.0.1] - 2026-06-20

Containerized deployment for a smooth, citation-enabled local test experience. No engine or safety
behaviour changed.

### Added

- **`Dockerfile` + `docker-compose.yml` + `.dockerignore`.** One-command run
  (`docker compose up --build` → `http://localhost:8000`) serving the API, the static UI, and the
  Proof artifacts from a single process.
- **RAG index built at image-build time**, with the local embedder (`bge-small-en-v1.5`) and
  cross-encoder reranker (`bge-reranker-base`) warmed into the image, so **cited SOAP observations
  work offline, out of the box** in the container. This is the recommended test surface:
  torch/sentence-transformers install cleanly on the Linux base, avoiding the Windows-host torch
  load failure that otherwise degrades retrieval to *uncited*.
- `HEALTHCHECK` on `/health`; CPU-only torch wheel to keep the image lean.

### Notes

- Secrets are **never** baked into the image. LLM credentials are read at runtime from `.env`
  (compose `env_file`); `.dockerignore` excludes `.env`, `data/`, and caches.
- `DEPLOY.md` updated with the containerized-run section.

## [1.0.0] - 2026-06-20

First complete release: an agentic pre-visit clinical intake chat with a **deterministic safety
gate**, cited RAG-grounded SOAP summaries, a published eval harness, and a release-acceptance gate.
Educational demo on **synthetic data only**, not a diagnostic, triage, or crisis service.

### The thesis

The safety guarantee comes from **code, upstream of the LLM**: a reproducible extractor and rule
engine run on every turn *before* the model. "0 missed on the frozen must-escalate set" is a literal
`assert` in CI, not "the model usually catches it." That is also the strongest prompt-injection
defense: there is no instruction for an injection to override.

### What's in 1.0.0

- **Scaffold, contracts & CI.** 4-module clean architecture (`core`/`api`/`eval`/`app`), Pydantic v2
  contracts, the two-tier (deterministic / live) test convention, and a per-commit CI gate.
- **Deterministic safety spine.** Pure regex/number extractor (no LLM), rule engine, and escalation
  templates; the must-escalate set is gated at 100% on every commit.
- **Orchestrator + agent loop.** The single per-turn code path; a tool-use loop with model routing
  and refusal handling. Wired to Azure GPT-5.5 (the spec pins Claude).
- **Intake engine + SOAP/triage.** Slot/state machine, adaptive branching, `build_summary`
  (schema-valid SOAP), and `suggest_triage` (band **clamped ≥ the safety floor**).
- **RAG subsystem.** Local hybrid retrieval (BM25 ∪ dense → cross-encoder rerank) over curated
  public-domain guidelines; a dependency-light local vector store; conservative citation binding.
- **Gold eval dataset.** 65 YAML scenarios including a **frozen** must-escalate set and held-out
  cases.
- **Eval harness + metrics + leaderboard.** Deterministic-gated metrics plus distributional metrics
  reported with spread; a byte-reproducible committed leaderboard.
- **LLM judge + retrieval evals + κ calibration.** Rubric judges (majority vote), a Cohen's κ
  meta-eval (judge-to-human), and RAGAS-style local retrieval metrics.
- **Observability, cost & prompt caching.** Cache-aware cost accounting (3 input buckets plus a
  counterfactual no-cache baseline), a latency report vs the targets, a live cache proof, and a
  dashboard.
- **FastAPI service.** A thin HTTP adapter over the in-process orchestrator; 5 routes plus SSE; the
  adapter carries no safety/model logic (grep-asserted).
- **Frontend wiring.** The finished `.dc.html` UI **connected, not rebuilt**, via a data-layer swap
  to the API; a Playwright browser smoke proves the backend is the safety authority.
- **Demo, README, leaderboard.** An honest two-group leaderboard, architecture diagrams, a
  compliance narrative, and a 2-minute demo script; every published number traces to an artifact.
- **Adversarial / prompt-injection red-team.** Injection cannot disable the gate (asserted,
  key-free); de-escalation cannot lower the monotonic floor; PII redaction for shareable exports;
  tracked residual rates reported, never gated.
- **Hardening, perf & release acceptance.** One repeatable acceptance run plus a **sign-off matrix**;
  the **six safety invariants re-asserted in one integration test**; a security/deps audit (secrets
  in tree + history, deps pinned, local-only paths); `DEPLOY.md` and the BAA-boundary posture; and a
  version bump plus release checklist.

### Safety invariants (held release-wide, re-asserted by `test_invariants_integration.py`)

1. The extractor contains no LLM/network call.
2. The gate runs in code upstream of the agent.
3. On a gate EMERGENCY the agent never runs (short-circuit).
4. The triage floor is monotonic per session.
5. Any safety-path exception fails safe (escalates, never silent-CLEAR).
6. The predicted band is never below the floor.

### Verified at release (deterministic, no API key)

- `pytest -m "not live"`: **637 passed**. `ruff check`: clean repo-wide.
- Deterministic eval gate: rule correctness **100%**, frozen must-escalate **0 miss**, triage floor
  never violated **100%**, schema validity **100%**.
- Audit: no committed secrets (tree + history), all deps pinned, safety+RAG paths local-only.
- Live (with a key): a routine turn drives GPT-5.5 → CLEAR question; emergency/injection
  short-circuit; intake latency p50 ≈ 1.7s / p95 ≈ 4.5s (within the 3s/6s targets).

[1.0.0]: https://example.com/scribeintake/releases/tag/v1.0.0

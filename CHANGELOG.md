# Changelog

All notable changes to ScribeIntake. This project follows a split-based build (14 self-contained
shippable units); the v1.0.0 entry summarizes the whole arc.

## [1.0.0] — 2026-06-20

First complete release: an agentic pre-visit clinical intake chat with a **deterministic safety
gate**, cited RAG-grounded SOAP summaries, a published eval harness, and a release-acceptance gate.
Educational demo on **synthetic data only** — not a diagnostic, triage, or crisis service.

### The thesis (what every split protects)
The safety guarantee comes from **code, upstream of the LLM** — a reproducible extractor → rule
engine runs on every turn *before* the model. "0 missed on the frozen must-escalate set" is a
literal `assert` in CI, not "the model usually catches it." That is also the strongest
prompt-injection defense: there is no instruction for an injection to override.

### Built across 14 splits
- **01 — Scaffold, contracts & CI.** 4-module clean architecture (`core`/`api`/`eval`/`app`),
  Pydantic v2 contracts, the two-tier (deterministic / live) test convention, per-commit CI gate.
- **02 — Deterministic safety spine.** Pure regex/number extractor (no LLM), rule engine,
  escalation templates; the must-escalate set gated at 100% on every commit.
- **03 — Orchestrator + agent loop.** The single per-turn code path (steps 1–7); tool-use loop with
  model routing + refusal handling. Wired to Azure GPT-5.5 (the env's key; the spec pins Claude).
- **04 — Intake engine + SOAP/triage.** Slot/state machine, adaptive branching, `build_summary`
  (schema-valid SOAP) + `suggest_triage` (band **clamped ≥ the safety floor**).
- **05 — RAG subsystem.** Local hybrid retrieval (BM25 ∪ dense → cross-encoder rerank) over
  curated public-domain guidelines; dependency-light local vector store; conservative citation binding.
- **06 — Gold eval dataset.** ~50→65 YAML scenarios incl. a **frozen** must-escalate set + held-out cases.
- **07 — Eval harness + metrics + leaderboard.** Deterministic-gated metrics + distributional
  metrics reported with spread; byte-reproducible committed leaderboard.
- **08 — LLM-judge + retrieval evals + κ calibration.** Rubric judges (majority vote), Cohen's κ
  meta-eval (judge↔human), RAGAS-style local retrieval metrics.
- **09 — Observability, cost & prompt caching.** Cache-aware cost accounting (3 input buckets +
  counterfactual no-cache baseline), latency report vs §18 targets, live cache proof, dashboard.
- **10 — FastAPI service.** Thin HTTP adapter over the in-process orchestrator; 5 routes + SSE; the
  adapter carries no safety/model logic (grep-asserted).
- **11 — Frontend wiring.** The finished `.dc.html` UI **connected, not rebuilt** — a data-layer
  swap to the API; Playwright browser smoke proves the backend is the safety authority.
- **12 — Demo, README, leaderboard, Loom.** Honest two-group leaderboard, architecture diagrams,
  compliance narrative, 2-min demo script; every published number traces to an artifact.
- **13 — Adversarial / prompt-injection red-team.** Injection cannot disable the gate (asserted,
  key-free); de-escalation cannot lower the monotonic floor; PII redaction for shareable exports;
  tracked residual rates reported, never gated.
- **14 — Hardening, perf & release acceptance.** One repeatable acceptance run + a **sign-off
  matrix**; the **six safety invariants re-asserted in one integration test**; security/deps audit
  (secrets in tree + history, deps pinned, local-only paths); `DEPLOY.md` + BAA-boundary posture;
  version bump + release checklist.

### Safety invariants (held release-wide, re-asserted by `test_invariants_integration.py`)
1. The extractor contains no LLM/network call. 2. The gate runs in code upstream of the agent.
3. On a gate EMERGENCY the agent never runs (short-circuit). 4. The triage floor is monotonic per
session. 5. Any safety-path exception fails safe (escalates, never silent-CLEAR). 6. The predicted
band is never below the floor.

### Verified at release (deterministic, no API key)
- `pytest -m "not live"`: **632 passed**. `ruff check`: clean repo-wide.
- Deterministic eval gate: rule correctness **100%**, frozen must-escalate **0 miss**, triage floor
  never violated **100%**, schema validity **100%**.
- Audit: no committed secrets (tree + history), all deps pinned, safety+RAG paths local-only.
- Live (with a key): routine turn drives GPT-5.5 → CLEAR question; emergency/injection short-circuit;
  intake latency p50 ≈ 1.7s / p95 ≈ 4.5s (within §18 3s/6s targets).

[1.0.0]: https://example.com/scribeintake/releases/tag/v1.0.0

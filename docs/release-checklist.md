# Release checklist (Split 14)

The human-runnable sign-off. A reviewer should be able to run **one command** and watch the system
pass. The acceptance script (`scripts/acceptance.py`) automates rows 1–8 and writes
[`acceptance_report.md`](../acceptance_report.md); this page is the manual wrapper + the BAA/release
steps that are a human action.

## One command

```bash
make acceptance-ci          # no API key — every gated row (suite, eval, invariants, audit, safety e2e)
make acceptance             # with a key in .env — adds the live model row + measured perf
cat acceptance_report.md    # the sign-off matrix
```

## The sign-off matrix (what acceptance proves)

Each safety invariant + each §18 target + each gated metric maps to a passing check
(`acceptance_report.md` → "Sign-off matrix"):

| # | Item | Proven by | Gated |
|---|---|---|---|
| 1 | INV-1 extractor has no LLM/network | `six_invariant_guard` | ✅ |
| 2 | INV-2 gate runs upstream of the agent | `six_invariant_guard` | ✅ |
| 3 | INV-3 EMERGENCY → agent never runs | `safety_e2e` | ✅ |
| 4 | INV-4 triage floor is monotonic | `six_invariant_guard` | ✅ |
| 5 | INV-5 safety-path exception fails safe | `six_invariant_guard` | ✅ |
| 6 | INV-6 predicted band ≥ floor (clamp) | `six_invariant_guard` | ✅ |
| 7 | GATE rule correctness 100% | `deterministic_eval` | ✅ |
| 8 | GATE frozen must-escalate 0-miss | `deterministic_eval` | ✅ |
| 9 | GATE triage floor never violated 100% | `deterministic_eval` | ✅ |
| 10 | GATE schema validity 100% | `deterministic_eval` | ✅ |
| 11 | §18 intake p50<3s / p95<6s | `perf` | informational |
| 12 | §17 no committed secrets / local-only | `audit:secrets` / `audit:local_only` | ✅ |
| 13 | CLAIM README numbers trace to artifacts | `docs_claims` | informational |

## Manual checklist

- [ ] **Pre-flight:** `PROGRESS.md` rows 01–13 all ✅ (release acceptance is meaningless over an
      incomplete spine).
- [ ] `make acceptance-ci` → **OVERALL: PASS**, no `FAIL` rows (no API key).
- [ ] `make acceptance` (key in `.env`) → the live model row **PASS**, perf within §18 (or every
      breach annotated with a remediation note).
- [ ] `python -m pytest core/tests/test_invariants_integration.py -v` → **26 passed** (the six
      invariants).
- [ ] `python scripts/audit.py` → secrets clean (tree + history), deps pinned, local-only.
- [ ] `ruff check` clean repo-wide.
- [ ] README leads with the architecture diagram + the honest two-group leaderboard; every headline
      number traces to an artifact; relative links resolve (`docs_claims`).
- [ ] `DEPLOY.md` (single-instance + BAA-boundary posture), `CHANGELOG.md`, this checklist exist and
      are consistent.
- [ ] `__version__` bumped; `PROGRESS.md` reconciled (all 14 ✅, decisions resolved, final log entry).

## Release (human action — only when asked)

- [ ] Commit the split.
- [ ] Tag: `git tag -a v1.0.0 -m "ScribeIntake v1.0.0"` and push the tag.
- [ ] For a HIPAA deployment, re-point the LLM client at a BAA-covered endpoint (Bedrock / Vertex /
      Claude platform) — see [`DEPLOY.md`](../DEPLOY.md) §3. Everything else is already local.

> Keep the "educational demo on synthetic data" framing through release: production-*ready* posture,
> demo-*scoped* data.

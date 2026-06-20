# ScribeIntake: Release Acceptance Report

- **Generated:** local-run
- **Overall:** ✅ PASS

## Sign-off matrix (each §18 target + safety invariant + gated metric → a passing check)

| Item | Status | Proven by |
|---|---|---|
| INV-1  extractor has no LLM/network | ✅ PASS | `six_invariant_guard` |
| INV-2  gate runs upstream of the agent | ✅ PASS | `six_invariant_guard` |
| INV-3  EMERGENCY → agent never runs | ✅ PASS | `safety_e2e` |
| INV-4  triage floor is monotonic | ✅ PASS | `six_invariant_guard` |
| INV-5  safety-path exception fails safe | ✅ PASS | `six_invariant_guard` |
| INV-6  predicted band >= floor (clamp) | ✅ PASS | `six_invariant_guard` |
| GATE   rule correctness 100% | ✅ PASS | `deterministic_eval` |
| GATE   frozen must-escalate 0-miss | ✅ PASS | `deterministic_eval` |
| GATE   triage floor never violated 100% | ✅ PASS | `deterministic_eval` |
| GATE   schema validity 100% | ✅ PASS | `deterministic_eval` |
| §18    intake latency p50<3s / p95<6s | ℹ️ INFO | `perf` |
| §17    no committed secrets / local-only | ✅ PASS | `audit:secrets` |
| §17    PHI paths local-only | ✅ PASS | `audit:local_only` |
| CLAIM  README numbers trace to artifacts | ✅ PASS | `docs_claims` |

## Checks

| Check | Status | Detail |
|---|---|---|
| `deterministic_suite` | ✅ PASS | 637 passed, 23 deselected in 12.35s |
| `deterministic_eval` | ✅ PASS | rule_correctness=100% · frozen_must_escalate=0 miss · triage_floor_never_violated=100% · schema_validity=100% |
| `six_invariant_guard` | ✅ PASS | 26 passed in 0.97s |
| `audit:secrets` | ✅ PASS | no key-shaped strings in the tracked tree; .env never committed; history clean |
| `audit:gitignored` | ✅ PASS | secrets/runtime-data/indexes gitignored (.env, data/, *.bin, .chroma) and untracked |
| `audit:deps_pinned` | ✅ PASS | all 13 declared deps carry a version bound |
| `audit:local_only` | ✅ PASS | safety + RAG paths are local-only (extractor/rules/retrieve/ingest); LLM is the only egress |
| `audit:pip_audit` | ℹ️ INFO | pip-audit not installed - scan skipped |
| `safety_e2e` | ✅ PASS | emergency → halt + agent never ran; injection → gate still fires EMERGENCY |
| `live_model_e2e` | ⏭️ SKIP | no API key (or --deterministic-only); live model row skipped |
| `perf` | ℹ️ INFO | no fresh traces; recorded live: intake p50=2290ms / p95=4715ms (both inside §18 3000/6000ms); re-measure with a keyed run |
| `docs_claims` | ✅ PASS | README headline numbers trace to leaderboard.json; all relative links resolve |


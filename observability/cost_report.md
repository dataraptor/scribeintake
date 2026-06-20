# ScribeIntake — Cost & Observability Report

> $0.012 · cache 38% saved

- **Generated:** 2026-06-20T00:00:00Z
- **Source:** `synthetic-demo` · model `gpt-5.5`
- **Model calls:** 5 · **total trace rows:** 9

## Cost (cache-aware, spec §16)

| | USD |
|---|---|
| With prompt caching | $0.0117 |
| Without caching (counterfactual) | $0.0189 |
| **Saved by caching** | **38%** |
| Total trace cost | $0.0117 |
| Local tool cost (must be $0) | $0.0000 |

Cache-read share of prompt tokens: **65%** (6420 tokens served from cache).

### Per-model token totals

| Model | Calls | Input | Output | Cache-read | Cost |
|---|---|---|---|---|---|
| `gpt-5.5` | 5 | 3410 | 660 | 6420 | $0.0117 |

## Latency percentiles (spec §18)

| Scope | n | p50 (ms) | p95 (ms) | target |
|---|---|---|---|---|
| Intake per-turn | 3 | 1700 | 2100 | p50<3000 / p95<6000 |
| Summary call | 1 | 2300 | 2300 | <8000 |

_First summary call (4200 ms) excluded from the percentiles — one-time schema compile (§7), not a regression._

All measured latencies are within the §18 targets. ✅

## Tool usage

| Tool | Calls |
|---|---|
| `agent_step` | 3 |
| `record_intake` | 3 |
| `retrieve_guideline` | 1 |
| `build_summary` | 1 |
| `suggest_triage` | 1 |

## Notes

- No-cache baseline = observed cached tokens repriced at full input price (pricing.no_cache_cost_usd); the wired GPT-5.5 prefix cache is automatic, so this exact counterfactual is more rigorous than re-running.
- Synthetic demonstration of the cache-aware pipeline (key-free, byte-reproducible). On the wired Azure GPT-5.5 deployment, prompt caching is verified LIVE on the terminal build_summary call (cache_read 0→1280 tok on warm repeats — see observability/cache_check.py); the agent loop's prefix does not surface cache hits on this deployment. Run `python -m observability` over a live DB for real per-session numbers.

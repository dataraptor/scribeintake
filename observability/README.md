# Observability, cost & prompt caching

A **read-only** analysis layer over the `tool_calls` audit table and the eval `runs/` JSONL: the
"reliability + cost control is what clients pay to keep" signal. It calls **no model** (except the
opt-in live cache check) and stands up **no service**; a notebook/static page is the point, not
another daemon.

## What's here

| Module | Purpose |
|---|---|
| `cost.py` | Cache-aware `$/session` from the **three input buckets**; per-model/per-tool breakdown; no-cache savings baseline; per-session JSONL export; `fleet_cost` over `runs/`. |
| `latency.py` | p50/p95 split into **intake per-turn** (target p50<3s/p95<6s) vs the **summary call** (~3–8s); the first-summary schema-compile is annotated, not flagged. |
| `cache_check.py` | **Live** cold-vs-warm prompt-cache proof (`cache_read` 0→>0 on warm repeats). |
| `report.py` / `dashboard.py` | `cost_report.{json,md}` + a static `dashboard.html` (consumed by the README and the frontend Proof tab). |

## Honest cost

Cost is computed from `input_tokens` (1.0×) + `cache_creation_input_tokens` (1.25×) +
`cache_read_input_tokens` (0.1×) + `output_tokens`. The **no-cache baseline** reprices the observed
cached tokens at full input price (`pricing.no_cache_cost_usd`), an exact counterfactual, more
rigorous than re-running. Local RAG rows are `$0`.

## Generate the report

```bash
python -m observability                 # over the live DB / eval runs/ (or a synthetic-demo)
make cost-report                        # same; tasks.ps1 cost-report on Windows
make cache-check                        # LIVE: cold-vs-warm cache_read proof (needs a key)
```

## Provider note (Azure GPT-5.5)

The wired provider does **automatic** prefix caching (no `cache_control`, no cache-write bucket;
`cache_read = prompt_tokens_details.cached_tokens`). Prompt caching is verified **live** on the
terminal `build_summary` call: `cache_read` rises **0 → 1280 tokens** on warm repeats (the large,
byte-stable `system + SOAP json_schema` prefix). The agent loop's smaller prefix does not surface
cache hits on this deployment; the cache-aware accounting and savings are proven deterministically
(`core/tests/test_cache_savings.py`, `observability/tests/`). The committed `cost_report.*` is a
key-free **synthetic demo** of the pipeline; live per-session numbers come from running the report
over a real DB.

# api

The backend service — a thin HTTP layer that exposes the engine over a network.

This layer is an adapter, not a place for business logic. It handles requests,
validation, configuration, and serialization, then delegates the real work to `core`.
If you deleted it, the engine would still work; you'd just lose the HTTP interface.

**Contains:** the web app and its routes, request/response schemas, configuration,
a container definition, and API-level tests.

**Depends on:** `core`.

## Run (Split 10)

```bash
pip install -e "./core[dev]" "fastapi>=0.110,<1" "uvicorn>=0.27,<1" "httpx>=0.27,<1"  # make install-api
python -m uvicorn api.main:app --port 8000                                            # make run-api
```

`api/` is import-only (resolved via the repo-root `pythonpath`); `.env` (Azure GPT-5.5 creds)
is read by `core` for live turns. Without creds the deterministic safety gate still runs — an
emergency message short-circuits with no model call.

## Endpoints (spec §14)

| Method | Path | Returns |
|---|---|---|
| `GET`  | `/health` | `{status, version, models}` |
| `POST` | `/session` | `{sessionId, disclaimer}` |
| `POST` | `/session/{id}/message` | **SSE** `token`* + terminal `turn` event (the `TurnResponse`). `Accept: application/json` → the same `TurnResponse` non-streamed. Unknown id → 404; model error → 503 / `error` frame, state preserved. |
| `GET`  | `/session/{id}/summary` | `SummaryResponse` (the SOAP) or 404 + reason when none |
| `GET`  | `/session/{id}/trace` | `TraceResponse` (rows + cache-aware totals) |

The API is a **thin adapter**: it calls `orchestrator.run_turn` / reads the DB and serialises the
result (`api/serialize.py`) into DTOs (`api/schemas.py`) whose field names match the `.dc.html`
frontend view-model (strip `ruleId`/`signalsView`/`toolsNote`; emergency `kicker`/`heading`/
`body`/`actions`/`caption`; summary `subjective`/`observations`/`red_flags_checked`; trace
`tool`/`model`/`cost`). No safety/intake/model logic lives here — the gate stays in `core`,
upstream. Each request is stateless (load → run → save; no in-memory session map).

# app — ScribeIntake frontend

The user-facing frontend. It **renders what the backend returns and handles interaction; it holds
no safety or business logic of its own.** In connected mode the deterministic safety gate, the
escalation floor, the emergency/crisis wording, and the SOAP summary all come from the API
(`core` upstream of the LLM) — the client only displays them.

## What's here

| File | Role |
|---|---|
| `ScribeIntake.dc.html` | The production UI — a finished React-ish [Design Component](https://en.wikipedia.org/wiki/Design_tool): an `<x-dc>` template + a `class Component extends DCLogic`. **The template + styling are the design source of truth and are not rebuilt.** Split 11 only swapped the *data-layer methods* inside the class. |
| `support.js` | The generated DC runtime: parses the `<x-dc>` template + the logic class and renders via React. Self-boots on load. |
| `api-client.js` | The **only** data layer (`window.SI_API`): the fetch/SSE client + the one place API JSON is renamed onto the component's view-model. |
| `index.html` | The host page: picks `API_BASE`, loads React + the client + the runtime, injects the component, boots it. |
| `vendor/` | Vendored React 18.3.1 UMD (offline-reproducible; the runtime CDN-loads it only if absent). |
| `tests/` | `test_api_client.mjs` (Node adapter/SSE unit tests, no browser) and `smoke_browser.py` (Playwright end-to-end against the real API). |

## Run it (connected — the one command)

The FastAPI service (Split 10) mounts this folder and serves the page same-origin:

```bash
# from the repo root, with the GPT-5.5 / model key in .env
python -m uvicorn api.main:app --port 8000      # or: make run-api
# open http://localhost:8000
```

That serves `index.html` at `/`, the component/client/React as static files, and the Proof
artifacts at `/proof/leaderboard.json` + `/proof/cost_report.json`. The frontend talks to the
same origin, so no CORS or `API_BASE` config is needed.

### Split-origin dev (static server + separate API)

Serve `app/` with any static server and point it at a separately-running API:

```bash
python -m uvicorn api.main:app --port 8000       # the API
python -m http.server 5500 --directory app       # the static frontend
# open http://localhost:5500/?api=http://localhost:8000
```

CORS defaults to `*` (set `API_CORS_ORIGINS` to pin it). `API_BASE` is resolved from, in order:
`window.API_BASE` → `?api=` query param → `<meta name="api-base">` → same-origin (`""`).

## Offline DEMO_MODE (the standalone mockup)

The component still runs with **no backend**, replaying its built-in scripted simulation
(`extract`/`gate`/`SCN`/…). This is automatic when `window.SI_API` is absent (opening
`ScribeIntake.dc.html` directly in the design tool), or forced with `?demo=1`:

```text
http://localhost:8000/?demo=1     # original simulation, zero API calls
```

In DEMO_MODE the client-side `RULES`/`extract` are the simulation only — **never the safety
authority when connected.** The connected path performs no safety decision.

## How the data swap works (connected mode)

- `handleSend` → `POST /session/{id}/message`, consumes the **SSE** `token`* stream into the
  assistant bubble, then applies the terminal `turn` event: the backend **strip** on the patient
  bubble, the floor, the slot progress (`openSlots`), the trace, and — on an EMERGENCY — the
  emergency sheet built from the **core template** (wording verbatim; the client only renders).
- `openSummary` → `GET /session/{id}/summary`, mapped into the existing summary sheet. If the
  intake isn't complete the API's honest reason is surfaced as a message (no fabricated SOAP).
- Proof tab → `GET /session/{id}/trace` (real cache-aware cost) + the real `leaderboard.json`
  (Split 07/08) and `cost_report.json` (Split 09).
- A network/stream error shows a friendly reconnect line and preserves the thread — never blank.

## Tests

```bash
node app/tests/test_api_client.mjs      # adapter + SSE-parser unit tests (no browser/network)
python app/tests/smoke_browser.py       # Playwright end-to-end against a booted API
```

The browser smoke's safety flows (emergency / crisis / injection) are **key-free** — the
deterministic gate short-circuits before any model call, proving the backend is the safety
authority. The routine streaming flow uses the live model key from `.env`.

**Depends on:** `api` (over HTTP/SSE, at runtime).

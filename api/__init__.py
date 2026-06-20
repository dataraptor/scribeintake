"""ScribeIntake HTTP API (Split 10) — a thin FastAPI adapter over the in-process engine.

This package contains **no business or safety logic**: it parses requests, calls
``scribeintake.orchestrator.run_turn`` / reads the SQLite DB, and serialises the result into
DTOs shaped for the frontend view-model. The safety gate, intake, triage and model calls all
live in ``core``; the API only exposes them over HTTP/SSE. Each request is stateless (load
session from SQLite, run, save) — there is no in-memory session map.
"""

__version__ = "1.0.0"

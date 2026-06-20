"""FastAPI app: the thin HTTP adapter over the in-process orchestrator (Split 10, spec §14/§6/§18).

Five routes — ``POST /session``, ``POST /session/{id}/message`` (SSE, with a JSON fallback),
``GET /session/{id}/summary``, ``GET /session/{id}/trace``, ``GET /health``. The app holds **no
business or safety logic**: each handler opens a fresh DB connection, calls
``orchestrator.run_turn`` / reads the DB, serialises via :mod:`api.serialize`, and closes the
connection. The safety gate runs in ``core`` upstream, exactly as before. Errors map to friendly
payloads — never a blank 500 (§18).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from scribeintake import db
from scribeintake.config import MODEL_SUMMARY, settings
from scribeintake.models import DISCLAIMER
from scribeintake.orchestrator import run_turn

from . import __version__, deps, schemas, serialize

# A friendly "try again" message for a model/transport failure mid-turn (§18). The patient's
# message is already persisted before any model call, so their state is preserved.
RECONNECT_MSG = (
    "We hit a snag reaching the assistant. Your information is saved — please send that again."
)
_STREAM_CHUNK = 24  # characters per streamed token frame (v1: chunk the final text, see §3.2)

# Repo root (parent of api/) — used to serve the static frontend and the committed proof artifacts.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_APP_DIR = _REPO_ROOT / "app"
# The Proof tab's real numbers (Split 07/08 leaderboard, Split 09 cost report). Served read-only.
_PROOF_FILES = {
    "leaderboard.json": _REPO_ROOT / "eval" / "leaderboard.json",
    "cost_report.json": _REPO_ROOT / "observability" / "cost_report.json",
}


def create_app(db_path: str | Path | None = None) -> FastAPI:
    """Build the FastAPI app. ``db_path`` overrides the SQLite path (tests pass a temp file)."""
    app = FastAPI(title="ScribeIntake API", version=__version__)
    app.state.db_path = Path(db_path) if db_path is not None else deps.default_db_path()

    # Ensure the schema exists once at startup (idempotent); handlers then open per-request conns.
    conn = db.connect(app.state.db_path)
    try:
        db.init_db(conn)
    finally:
        conn.close()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=deps.cors_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _register_routes(app)
    _mount_frontend(app)
    return app


# --------------------------------------------------------------------------------- SSE helpers
def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _chunks(text: str, size: int = _STREAM_CHUNK):
    for i in range(0, len(text), size):
        yield text[i : i + size]


def _not_found(session_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content=schemas.ErrorResponse(
            error="session_not_found",
            detail=f"No session with id {session_id!r}.",
        ).model_dump(by_alias=True),
    )


def _no_summary_reason(session: Any) -> str:
    status = session["status"] if session is not None else "active"
    if status == "halted":
        return "This session ended with a safety referral and produced no clinical summary."
    return "No summary yet — the intake is not complete."


# ------------------------------------------------------------------------------------- routes
def _register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=schemas.HealthResponse)
    def health() -> schemas.HealthResponse:
        return schemas.HealthResponse(
            version=__version__,
            models={"intake": settings.ACTIVE_INTAKE_MODEL, "summary": MODEL_SUMMARY},
        )

    @app.post("/session", response_model=schemas.StartSessionResponse)
    def start_session(request: Request) -> schemas.StartSessionResponse:
        conn = deps.open_conn(request.app.state.db_path)
        try:
            session_id = db.create_session(conn)
        finally:
            conn.close()
        return schemas.StartSessionResponse(session_id=session_id, disclaimer=DISCLAIMER)

    @app.post("/session/{session_id}/message")
    def message(session_id: str, body: schemas.MessageRequest, request: Request):
        db_path = request.app.state.db_path

        # Validate the session up front so an unknown id is a clean 404, not a mid-stream error.
        conn = deps.open_conn(db_path)
        try:
            exists = db.get_session(conn, session_id) is not None
        finally:
            conn.close()
        if not exists:
            return _not_found(session_id)

        accept = request.headers.get("accept", "")
        wants_json = "application/json" in accept and "text/event-stream" not in accept

        if wants_json:
            conn = deps.open_conn(db_path)
            try:
                turn = run_turn(session_id, body.text, conn=conn)
            except Exception:  # noqa: BLE001 - any model/transport error -> friendly reconnect
                return JSONResponse(
                    status_code=503,
                    content=schemas.ErrorResponse(
                        error="upstream_unavailable", detail=RECONNECT_MSG
                    ).model_dump(by_alias=True),
                )
            finally:
                conn.close()
            return JSONResponse(serialize.turn_response(turn).model_dump(by_alias=True))

        def event_stream():
            conn = deps.open_conn(db_path)
            try:
                turn = run_turn(session_id, body.text, conn=conn)
            except Exception:  # noqa: BLE001 - never drop the stream blankly (§18)
                conn.close()
                yield _sse("error", {"message": RECONNECT_MSG, "kind": "reconnect"})
                return
            conn.close()
            resp = serialize.turn_response(turn)
            for chunk in _chunks(resp.content):
                yield _sse("token", {"text": chunk})
            yield _sse("turn", resp.model_dump(by_alias=True))

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/session/{session_id}/summary")
    def summary(session_id: str, request: Request):
        conn = deps.open_conn(request.app.state.db_path)
        try:
            session = db.get_session(conn, session_id)
            if session is None:
                return _not_found(session_id)
            row = db.get_summary(conn, session_id)
            band = session["triage_band"] if session is not None else None
        finally:
            conn.close()
        if row is None:
            return JSONResponse(
                status_code=404,
                content=schemas.ErrorResponse(
                    error="no_summary", detail=_no_summary_reason(session)
                ).model_dump(by_alias=True),
            )
        soap = json.loads(row["soap_json"])
        return JSONResponse(serialize.summary_response(soap, band=band).model_dump(by_alias=True))

    @app.get("/proof/{name}")
    def proof(name: str):
        """Serve a committed Proof artifact (leaderboard / cost report) for the Proof tab.

        Read-only static JSON — no engine or model logic. Missing file -> a friendly 404 so the
        frontend simply falls back to its placeholders.
        """
        path = _PROOF_FILES.get(name)
        if path is None or not path.exists():
            return JSONResponse(
                status_code=404,
                content=schemas.ErrorResponse(
                    error="proof_not_found", detail=f"No proof artifact {name!r}."
                ).model_dump(by_alias=True),
            )
        return FileResponse(path, media_type="application/json")

    @app.get("/session/{session_id}/trace")
    def trace(session_id: str, request: Request):
        conn = deps.open_conn(request.app.state.db_path)
        try:
            if db.get_session(conn, session_id) is None:
                return _not_found(session_id)
            rows = db.get_tool_calls(conn, session_id)
        finally:
            conn.close()
        return JSONResponse(serialize.trace_response(rows, session_id).model_dump(by_alias=True))


def _mount_frontend(app: FastAPI) -> None:
    """Mount the static frontend at ``/`` so ``uvicorn api.main:app`` serves the whole app.

    This is **serving wiring only** (the Split-11 one-command serve, spec §14) — no business or
    safety logic. ``html=True`` serves ``app/index.html`` at ``/`` and the component / client /
    vendored React as static files. Mounted last so the explicit API + ``/proof`` routes win.
    """
    if _APP_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(_APP_DIR), html=True), name="frontend")


# The module-level app for ``uvicorn api.main:app`` (uses the default data/ DB path).
app = create_app()

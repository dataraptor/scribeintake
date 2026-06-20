"""Drive one gold scenario end-to-end through the real per-turn pipeline (spec section 15).

The harness **imports the orchestrator directly** (:func:`scribeintake.orchestrator.run_turn`)
and runs entirely in-process — never over the network. That is what keeps eval runs fast,
isolated, and parallel-safe: every :func:`run_scenario` gets a **fresh SQLite database**, feeds
the scenario's turns through the same code path the production app uses, captures a
:class:`~eval.models.TurnRecord` per turn, and tears the database down. Because the orchestrator
is stateless per turn (state lives only in SQLite, spec section 6), two scenarios can never bleed
state into each other.

The same entry point serves both tiers: tests inject a mocked ``agent`` / ``summary_client``
(no key), while the live runner passes the real clients built once and reused across runs.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile

from scribeintake import db
from scribeintake.models import TriageBand
from scribeintake.orchestrator import run_turn

from .models import ScenarioRun, TurnRecord
from .scenario import Scenario

# A fixed timestamp stamped into every SOAP so runs are byte-reproducible across machines and
# repeats (the orchestrator passes this straight into ``generated_at``; spec section 3.4).
EVAL_GENERATED_AT = "2026-01-01T00:00:00+00:00"


def run_scenario(
    scenario: Scenario,
    *,
    seed_label: str,
    db_path: str | None = None,
    agent: object | None = None,
    summary_client: object | None = None,
    retriever: object | None = None,
    generated_at: str = EVAL_GENERATED_AT,
) -> ScenarioRun:
    """Run one ``scenario`` end-to-end and return a hermetic :class:`ScenarioRun`.

    Each call owns its database: a fresh, isolated SQLite file (a temp file unless ``db_path``
    is given) created, used, and removed here. Turns are fed in order; if the session **halts**
    (an EMERGENCY short-circuit), the remaining turns are **not** fed — an emergency intake does
    not continue — and ``intake_halted`` is recorded.

    ``agent`` / ``summary_client`` / ``retriever`` are passed straight to the orchestrator. The
    deterministic tier injects fakes (no key); the live tier passes real clients built once and
    reused (safe — the orchestrator holds no state between turns). ``seed_label`` distinguishes
    the N repeats of the same scenario.
    """
    own_db = db_path is None
    if own_db:
        fd, db_path = tempfile.mkstemp(suffix=".db", prefix="scribeintake_eval_")
        os.close(fd)

    conn = db.reset_db(db_path)
    try:
        return _run(
            scenario,
            seed_label=seed_label,
            conn=conn,
            agent=agent,
            summary_client=summary_client,
            retriever=retriever,
            generated_at=generated_at,
        )
    finally:
        conn.close()
        if own_db:
            _unlink(db_path)


def _run(
    scenario: Scenario,
    *,
    seed_label: str,
    conn: sqlite3.Connection,
    agent: object | None,
    summary_client: object | None,
    retriever: object | None,
    generated_at: str,
) -> ScenarioRun:
    session_id = db.create_session(conn)
    turns: list[TurnRecord] = []
    halted = False
    last_safety_id = 0
    final_status = "active"

    for user_msg in scenario.turns:
        result = run_turn(
            session_id,
            user_msg,
            conn=conn,
            agent=agent,
            summary_client=summary_client,
            retriever=retriever,
            generated_at=generated_at,
        )
        new_events, last_safety_id = _new_safety_events(conn, session_id, last_safety_id)
        turns.append(_to_turn_record(result, user_msg, new_events))
        final_status = result.status
        if result.status == "halted":
            halted = True
            break  # emergency short-circuit — do not keep interviewing

    # Load the *persisted* SOAP + band (proves the finalization wrote them; §3.1 step 4).
    final_soap, predicted_band = (
        _load_final(conn, session_id) if final_status == "completed" else (None, None)
    )

    return ScenarioRun(
        scenario_id=scenario.id,
        category=scenario.category,
        heldout=scenario.heldout,
        seed_label=seed_label,
        turns=turns,
        final_soap=final_soap,
        predicted_band=predicted_band,
        intake_halted=halted,
        final_status=final_status,
        n_turns_run=len(turns),
        total_cost_usd=sum(t.cost_usd for t in turns),
        total_latency_ms=sum(t.latency_ms for t in turns),
        total_input_tokens=sum(t.input_tokens for t in turns),
        total_output_tokens=sum(t.output_tokens for t in turns),
    )


def _to_turn_record(result: object, user_msg: str, safety_events: list[dict]) -> TurnRecord:
    """Fold one :class:`~scribeintake.orchestrator.AssistantTurn` into a persisted record."""
    traces = result.traces  # type: ignore[attr-defined]
    return TurnRecord(
        turn=result.turn,  # type: ignore[attr-defined]
        user_msg=user_msg,
        escalation=result.level,  # type: ignore[attr-defined]
        escalation_source=result.source,  # type: ignore[attr-defined]
        matched_rules=list(result.matched_rules),  # type: ignore[attr-defined]
        crisis=result.crisis,  # type: ignore[attr-defined]
        triage_floor=result.triage_floor,  # type: ignore[attr-defined]
        status=result.status,  # type: ignore[attr-defined]
        assistant_text=result.assistant_text,  # type: ignore[attr-defined]
        tools_used=list(result.tools_used),  # type: ignore[attr-defined]
        safety_events=safety_events,
        input_tokens=sum(t.input_tokens for t in traces),
        output_tokens=sum(t.output_tokens for t in traces),
        cost_usd=sum(t.cost_usd for t in traces),
        latency_ms=sum((t.latency_ms or 0) for t in traces),
        model=result.model,  # type: ignore[attr-defined]
    )


def _new_safety_events(
    conn: sqlite3.Connection, session_id: str, after_id: int
) -> tuple[list[dict], int]:
    """Return the ``safety_events`` rows written since ``after_id`` and the new high-water id."""
    rows = conn.execute(
        "SELECT id, level, source FROM safety_events "
        "WHERE session_id = ? AND id > ? ORDER BY id ASC",
        (session_id, after_id),
    ).fetchall()
    events = [{"level": r["level"], "source": r["source"]} for r in rows]
    high = rows[-1]["id"] if rows else after_id
    return events, high


def _load_final(conn: sqlite3.Connection, session_id: str) -> tuple[dict | None, TriageBand | None]:
    """Load the persisted final SOAP (latest ``summaries`` row) + the session's final band."""
    import json

    row = conn.execute(
        "SELECT soap_json FROM summaries WHERE session_id = ? ORDER BY id DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    soap = json.loads(row["soap_json"]) if row else None

    srow = db.get_session(conn, session_id)
    band = TriageBand(srow["triage_band"]) if srow and srow["triage_band"] else None
    return soap, band


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass

"""Shared fixtures for the API suite (Split 10).

Provides a :class:`TestClient` over an app bound to an isolated temp SQLite DB, plus builders for
the :class:`~scribeintake.orchestrator.AssistantTurn` objects the serialize/endpoint tests feed in
(the orchestrator is **mocked** in the deterministic tier — no model call, no API key).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from scribeintake.models import (
    EscalationLevel,
    EscalationSource,
    Signals,
    ToolCallTrace,
    TriageBand,
)
from scribeintake.orchestrator import AssistantTurn
from scribeintake.safety import crisis_template, emergency_template


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "api_test.db"


@pytest.fixture
def app(db_path):
    return create_app(db_path)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def session_id(client) -> str:
    """A real session created through the API (so message/summary/trace have a valid id)."""
    resp = client.post("/session")
    return resp.json()["sessionId"]


# ----------------------------------------------------------------------- AssistantTurn builders
def make_clear_turn(
    session_id: str = "sess1", *, assistant_text: str = "How long has this been going on?"
) -> AssistantTurn:
    """A normal CLEAR intake turn with an agent question + a couple of trace rows."""
    traces = [
        ToolCallTrace(
            session_id=session_id,
            turn=1,
            tool="agent_step",
            model="gpt-5.5",
            input_tokens=1200,
            output_tokens=80,
            cache_read_tokens=0,
            latency_ms=900,
            cost_usd=0.0023,
            result_json='{"text": "...", "tool_calls": ["record_intake"]}',
        ),
        ToolCallTrace(
            session_id=session_id,
            turn=1,
            tool="record_intake",
            model=None,
            cost_usd=0.0,
            latency_ms=1,
            result_json='{"open_slots": ["hpi.onset"], "branch_hints": []}',
        ),
        ToolCallTrace(
            session_id=session_id,
            turn=1,
            tool="retrieve_guideline",
            model=None,
            cost_usd=0.0,
            latency_ms=5,
            result_json=(
                '{"chunks": [{"chunk_id": "chk_abc123", "text": "...", '
                '"source": "MedlinePlus", "url": "https://medlineplus.gov/x", "score": 0.81}]}'
            ),
        ),
    ]
    return AssistantTurn(
        session_id=session_id,
        turn=1,
        assistant_text=assistant_text,
        level=EscalationLevel.CLEAR,
        source=EscalationSource.gate,
        matched_rules=[],
        crisis=False,
        triage_floor=TriageBand.self_care,
        status="active",
        template=None,
        signals=Signals(dyspnea=True, sbp=None).model_dump(),
        open_slots=["hpi.onset", "hpi.severity"],
        tools_used=["record_intake", "retrieve_guideline"],
        traces=traces,
        model="gpt-5.5",
    )


def make_gate_emergency_turn(session_id: str = "sess1") -> AssistantTurn:
    """A deterministic gate EMERGENCY (ACS) — agent never ran, status halted."""
    tmpl = emergency_template()
    return AssistantTurn(
        session_id=session_id,
        turn=1,
        assistant_text=f"{tmpl['heading']}\n\n{tmpl['body']}",
        level=EscalationLevel.EMERGENCY,
        source=EscalationSource.gate,
        matched_rules=["acs_chest_pain"],
        crisis=False,
        triage_floor=TriageBand.ER,
        status="halted",
        template=tmpl,
        signals=Signals(chest_pain=True, diaphoresis=True).model_dump(),
        open_slots=[],
        tools_used=[],
        traces=[],
        model=None,
    )


def make_agent_emergency_turn(session_id: str = "sess1") -> AssistantTurn:
    """An agent-net EMERGENCY (oblique phrasing the regex missed) — second-net catch."""
    tmpl = emergency_template()
    return AssistantTurn(
        session_id=session_id,
        turn=2,
        assistant_text=f"{tmpl['heading']}\n\n{tmpl['body']}",
        level=EscalationLevel.EMERGENCY,
        source=EscalationSource.agent,
        matched_rules=["agent_assessment"],
        crisis=False,
        triage_floor=TriageBand.ER,
        status="halted",
        template=tmpl,
        signals=Signals().model_dump(),
        open_slots=[],
        tools_used=["assess_escalation"],
        traces=[],
        model="gpt-5.5",
    )


def make_crisis_turn(session_id: str = "sess1") -> AssistantTurn:
    """A mental-health crisis EMERGENCY routed to the compassionate template."""
    tmpl = crisis_template()
    return AssistantTurn(
        session_id=session_id,
        turn=1,
        assistant_text=f"{tmpl['heading']}\n\n{tmpl['body']}",
        level=EscalationLevel.EMERGENCY,
        source=EscalationSource.gate,
        matched_rules=["suicidal_crisis"],
        crisis=True,
        triage_floor=TriageBand.ER,
        status="halted",
        template=tmpl,
        signals=Signals(suicidal_ideation=True).model_dump(),
        open_slots=[],
        tools_used=[],
        traces=[],
        model=None,
    )


def make_completed_turn(session_id: str = "sess1") -> AssistantTurn:
    """A completion turn carrying a finalized SOAP + clamped band."""
    soap = sample_soap()
    return AssistantTurn(
        session_id=session_id,
        turn=4,
        assistant_text="Thanks — I've prepared a summary for your clinician.",
        level=EscalationLevel.CLEAR,
        source=EscalationSource.gate,
        matched_rules=[],
        crisis=False,
        triage_floor=TriageBand.self_care,
        status="completed",
        template=None,
        signals=Signals().model_dump(),
        open_slots=[],
        tools_used=["record_intake"],
        traces=[
            ToolCallTrace(
                session_id=session_id,
                turn=4,
                tool="build_summary",
                model="gpt-5.5",
                input_tokens=1800,
                output_tokens=320,
                cost_usd=0.0055,
                latency_ms=4200,
            )
        ],
        model="gpt-5.5",
        soap=soap,
        triage_band=TriageBand.gp_routine,
    )


def sample_soap() -> dict:
    """A representative persisted SOAP dict (one cited, one uncited observation)."""
    return {
        "subjective": {
            "chief_complaint": "dull headache",
            "hpi": {
                "onset": "two days ago",
                "severity": "4 out of 10",
                "radiation": "none",
                "character": "dull",
                "location": "",
                "duration": "",
                "aggravating": "",
                "relieving": "",
                "timing": "",
            },
            "medications": ["ibuprofen as needed"],
            "allergies": ["penicillin"],
            "past_history": [],
            "social": "",
            "low_confidence_fields": ["hpi.severity"],
        },
        "objective": {
            "patient_reported_vitals": {
                "sbp": "186",
                "dbp": "122",
                "glucose_mgdl": None,
                "spo2": None,
                "hr": None,
                "temp_f": None,
            },
            "notes": "",
        },
        "observations": [
            {
                "text": "Same-day clinician evaluation advised for elevated home blood pressure.",
                "citation": {
                    "source": "MedlinePlus",
                    "url": "https://medlineplus.gov/highbloodpressure.html",
                    "chunk_id": "chk_bp01",
                },
            },
            {"text": "No red-flag features triggered this session.", "citation": None},
        ],
        "triage": {
            "band": "gp_routine",
            "rationale": "Elevated home BP without acute symptoms warrants routine review.",
            "citations": [],
        },
        "red_flags_checked": [f"rule_{i}" for i in range(21)],
        "red_flags_triggered": [],
        "generated_at": "2026-06-20T12:00:00+00:00",
        "disclaimer": "Not a diagnosis. For clinician review.",
    }

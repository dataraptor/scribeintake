"""Finalization wiring (Split 04 §3.6, acceptance #1/#5/#6) — Opus call mocked.

With ``build_summary``/``suggest_triage`` backed by a scripted structured client (no network),
a completed intake persists a ``summaries`` row, sets ``status="completed"``, writes two
terminal trace rows, and the SOAP's triage band equals the clamped band. An emergency yields
no summary. Also covers ``build_summary``'s deterministic stamping + refusal/backstop paths.
"""

from __future__ import annotations

from fakes import FakeLLMClient, FakeStructuredClient, text_response, tool_response

from scribeintake import db
from scribeintake.agent import AgentLoop
from scribeintake.models import SOAP, Confidence, Subjective, TriageBand
from scribeintake.orchestrator import run_turn
from scribeintake.safety.rules import RULES
from scribeintake.tools import default_registry
from scribeintake.tools.build_summary import build_summary
from scribeintake.tools.suggest_triage import TriageSuggestion, suggest_triage

CLEAR_MSG = "I've had a mild sore throat for a couple of days."
URGENT_MSG = "I have severe abdominal pain that is really bad."
EMERGENCY_MSG = "I have crushing chest pain spreading to my left arm and I'm sweating."


# ------------------------------------------------------------------- helpers
def _fill_agent(updates, question="Anything else?") -> AgentLoop:
    ups = [u if isinstance(u, dict) else {"slot": u[0], "value": u[1]} for u in updates]
    return AgentLoop(
        FakeLLMClient(
            [tool_response([("record_intake", {"updates": ups})]), text_response(question)]
        ),
        default_registry(),
    )


GENERIC_FILL = [
    ("chief_complaint", "sore throat"),
    ("hpi.onset", "2 days ago"),
    ("hpi.severity", "mild"),
    ("medications", "none"),
    ("allergies", "none"),
]
ABDO_FILL = [
    ("chief_complaint", "severe abdominal pain"),
    ("hpi.onset", "today"),
    ("hpi.severity", "8/10"),
    ("hpi.location", "lower right"),
    ("medications", "none"),
    ("allergies", "none"),
]


def _summary_client(
    band: TriageBand = TriageBand.self_care, rationale="benign"
) -> FakeStructuredClient:
    return FakeStructuredClient(
        {
            "SOAP": SOAP(subjective=Subjective(chief_complaint="(model)")),
            "TriageSuggestion": TriageSuggestion(band=band, rationale=rationale),
        }
    )


def _summaries(conn, session):
    return conn.execute(
        "SELECT soap_json, version FROM summaries WHERE session_id = ?", (session,)
    ).fetchall()


# ------------------------------------------------------- full finalize (CLEAR)
def test_completed_intake_persists_summary_and_traces(conn, session):
    client = _summary_client(TriageBand.self_care)
    turn = run_turn(
        session, CLEAR_MSG, conn=conn, agent=_fill_agent(GENERIC_FILL), summary_client=client
    )

    assert turn.status == "completed"
    assert turn.open_slots == []
    assert turn.soap is not None
    assert turn.triage_band is TriageBand.self_care

    # one summaries row, soap_json parses, triage band matches the clamped band
    rows = _summaries(conn, session)
    assert len(rows) == 1
    soap = SOAP.model_validate_json(rows[0]["soap_json"])
    assert soap.triage.band is TriageBand.self_care
    # red_flags_checked is the LIVE rule count, never a hardcoded 18/20
    assert len(soap.red_flags_checked) == len(RULES)

    # session marked completed with a final band + completed_at
    srow = db.get_session(conn, session)
    assert srow["status"] == "completed"
    assert srow["triage_band"] == "self_care"
    assert srow["completed_at"]

    # two terminal trace rows, with model + non-zero cost
    fin = conn.execute(
        "SELECT tool, model, cost_usd FROM tool_calls "
        "WHERE session_id = ? AND tool IN ('build_summary','suggest_triage') ORDER BY id",
        (session,),
    ).fetchall()
    assert [r["tool"] for r in fin] == ["build_summary", "suggest_triage"]
    assert all(r["model"] and r["cost_usd"] > 0 for r in fin)


# ------------------------------------------------ clamp through the orchestrator
def test_finalize_clamps_band_up_to_safety_floor(conn, session):
    # URGENT gate pins floor gp_urgent; the model suggesting self_care must be clamped up.
    client = _summary_client(TriageBand.self_care)
    turn = run_turn(
        session, URGENT_MSG, conn=conn, agent=_fill_agent(ABDO_FILL), summary_client=client
    )
    assert turn.status == "completed"
    assert turn.triage_floor is TriageBand.gp_urgent
    assert turn.triage_band is TriageBand.gp_urgent  # clamped, not self_care
    soap = SOAP.model_validate_json(_summaries(conn, session)[0]["soap_json"])
    assert soap.triage.band is TriageBand.gp_urgent
    assert db.get_session(conn, session)["triage_band"] == "gp_urgent"


# ---------------------------------------------- emergency yields NO summary (#6)
def test_emergency_short_circuit_produces_no_summary(conn, session):
    client = _summary_client()
    turn = run_turn(
        session, EMERGENCY_MSG, conn=conn, agent=_fill_agent(GENERIC_FILL), summary_client=client
    )
    assert turn.status == "halted"
    assert turn.soap is None
    assert _summaries(conn, session) == []  # no SOAP for an emergency
    assert client.calls == []  # the terminal calls never ran


# --------------------------------------------------- low-confidence propagation
def test_low_confidence_fields_populated_on_finalize(conn, session):
    client = _summary_client()
    agent = _fill_agent(
        [
            {"slot": "chief_complaint", "value": "sore throat", "confidence": "high"},
            {"slot": "hpi.onset", "value": "2 days ago", "confidence": "high"},
            {"slot": "hpi.severity", "value": "maybe a 6", "confidence": "medium"},
            {"slot": "medications", "value": "not sure", "confidence": "unknown"},
            {"slot": "allergies", "value": "none", "confidence": "high"},
        ]
    )
    turn = run_turn(session, CLEAR_MSG, conn=conn, agent=agent, summary_client=client)
    low = turn.soap["subjective"]["low_confidence_fields"]
    assert "hpi.severity" in low  # medium
    assert "medications" in low  # unknown
    assert "allergies" not in low  # high


# ------------------------------------------- build_summary unit (stamp/refusal/backstop)
def _state_with(**slots):
    from scribeintake.models import IntakeState, SlotValue

    st = IntakeState(session_id="s")
    for k, v in slots.items():
        key = k.replace("__", ".")
        st.slots[key] = SlotValue(value=v, confidence=Confidence.high)
    return st


def test_build_summary_stamps_deterministic_fields():
    st = _state_with(chief_complaint="sore throat")
    client = FakeStructuredClient({"SOAP": SOAP(subjective=Subjective(chief_complaint="x"))})
    res = build_summary(st, client=client, generated_at="2026-06-20T00:00:00Z")
    assert res.refused is False
    assert res.soap.generated_at == "2026-06-20T00:00:00Z"
    assert res.soap.disclaimer  # the DISCLAIMER constant
    assert len(res.soap.red_flags_checked) == len(RULES)
    # effort routed high for the quality-critical summary call
    assert client.calls[0]["effort"] == "high"


def test_build_summary_refusal_falls_back_to_safe_shell():
    st = _state_with(chief_complaint="headache")
    client = FakeStructuredClient({"SOAP": SOAP()}, refuse=True)
    res = build_summary(st, client=client, generated_at="t")
    assert res.refused is True
    # honest shell: chief complaint preserved, no invented prose, metadata still stamped
    assert res.soap.subjective.chief_complaint == "headache"
    assert len(res.soap.red_flags_checked) == len(RULES)


def test_build_summary_retries_once_on_max_tokens():
    st = _state_with(chief_complaint="cough")
    client = FakeStructuredClient(
        {"SOAP": SOAP(subjective=Subjective(chief_complaint="cough"))},
        stop_reasons=["max_tokens", "end_turn"],
    )
    res = build_summary(st, client=client, generated_at="t")
    assert len(client.calls) == 2  # one backstop retry
    assert client.calls[1]["max_tokens"] > client.calls[0]["max_tokens"]
    assert res.refused is False


# ------------------------------------------- suggest_triage unit (clamp via client)
def test_suggest_triage_clamps_model_band_to_floor():
    st = _state_with(chief_complaint="chest pain")
    soap = SOAP(subjective=Subjective(chief_complaint="chest pain"))
    client = FakeStructuredClient({"TriageSuggestion": TriageSuggestion(band=TriageBand.self_care)})
    res = suggest_triage(st, soap, floor=TriageBand.gp_urgent, client=client)
    assert res.model_band is TriageBand.self_care  # what the model said
    assert res.triage.band is TriageBand.gp_urgent  # what code enforced
    assert res.triage.citations == []  # empty until Split 05

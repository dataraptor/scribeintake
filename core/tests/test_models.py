"""Deterministic tests for the Pydantic contracts.

The headline guard: ``SOAP.model_json_schema()`` must be native-structured-output-safe
(spec section 12) so Split 04 can hand it straight to ``output_config.format``.
"""

import json

from scribeintake import models as m

_BANNED_SCHEMA_KEYS = (
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "pattern",
    "multipleOf",
)


def test_enum_members_exact():
    assert {e.value for e in m.Confidence} == {"high", "medium", "unknown"}
    assert {e.value for e in m.EscalationLevel} == {"CLEAR", "URGENT", "EMERGENCY"}
    assert {e.value for e in m.TriageBand} == {"self_care", "gp_routine", "gp_urgent", "ER"}
    assert {e.value for e in m.EscalationSource} == {"gate", "agent"}


def _iter_objects(node):
    """Yield every JSON-schema node whose ``type`` is ``object``."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for value in node.values():
            yield from _iter_objects(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_objects(item)


def test_soap_schema_is_structured_output_safe():
    schema = m.SOAP.model_json_schema()
    blob = json.dumps(schema)

    # No unsupported JSON-schema constructs anywhere in the schema.
    for banned in _BANNED_SCHEMA_KEYS:
        assert banned not in blob, f"unsupported schema key present: {banned}"

    # additionalProperties:false on every object (root + every $def).
    objects = list(_iter_objects(schema))
    assert objects, "expected at least one object in the SOAP schema"
    for obj in objects:
        title = obj.get("title")
        assert obj.get("additionalProperties") is False, (
            f"object missing additionalProperties:false: {title}"
        )


def test_soap_round_trips():
    soap = m.SOAP(
        subjective=m.Subjective(
            chief_complaint="chest tightness, exertional",
            hpi=m.HPI(onset="this morning", severity="4/10"),
            medications=["lisinopril"],
            allergies=["none reported"],
            low_confidence_fields=["hpi.severity"],
        ),
        objective=m.Objective(
            patient_reported_vitals=m.PatientReportedVitals(sbp="186", dbp="122"),
            notes="home monitor",
        ),
        observations=[
            m.Observation(
                text="Same-day clinician evaluation advised.",
                citation=m.Citation(
                    source="MedlinePlus",
                    url="https://medlineplus.gov/heartattack.html",
                    chunk_id="chk_0142",
                ),
            ),
            m.Observation(text="Red flags screened; none triggered."),
        ],
        triage=m.Triage(
            band=m.TriageBand.gp_urgent,
            rationale="new exertional chest discomfort",
            citations=[m.Citation(source="MedlinePlus", chunk_id="chk_0142")],
        ),
        red_flags_checked=["acs_chest_pain", "stroke_fast"],
        red_flags_triggered=[],
        generated_at="2026-06-20T00:00:00+00:00",
    )
    dumped = soap.model_dump()
    assert m.SOAP(**dumped).model_dump() == dumped
    assert dumped["disclaimer"] == m.DISCLAIMER


def test_default_soap_instantiates():
    soap = m.SOAP()
    assert soap.disclaimer == m.DISCLAIMER
    assert soap.triage.band == m.TriageBand.self_care
    assert soap.red_flags_checked == []


def test_signals_defaults():
    s = m.Signals()
    dumped = s.model_dump()

    booleans = {k: v for k, v in dumped.items() if isinstance(v, bool)}
    assert booleans, "expected boolean symptom flags"
    assert all(v is False for v in booleans.values())

    for numeric in ("sbp", "dbp", "glucose_mgdl", "spo2", "hr", "temp_f"):
        assert dumped[numeric] is None


def test_tool_io_shapes():
    out = m.RecordIntakeOutput(open_slots=["allergies"], branch_hints=["radiation"])
    assert out.open_slots == ["allergies"]

    chunk = m.RetrievedChunk(chunk_id="chk_1", text="t", source="MedlinePlus", url="u", score=0.88)
    assert chunk.score == 0.88

    rec = m.RecordIntakeInput(
        updates=[m.SlotUpdate(slot="medications", value="none", confidence=m.Confidence.high)]
    )
    assert rec.updates[0].confidence is m.Confidence.high

    esc = m.AssessEscalationInput(
        level=m.EscalationLevel.EMERGENCY, rationale="left arm pain + sweating"
    )
    assert esc.level is m.EscalationLevel.EMERGENCY


def test_tool_call_trace_defaults_versions():
    trace = m.ToolCallTrace(session_id="s1", tool="retrieve_guideline", model="claude-sonnet-4-6")
    assert trace.prompt_version == "v1"
    assert trace.rules_version == "v1"
    assert trace.cost_usd == 0.0
    assert trace.model == "claude-sonnet-4-6"

"""Live structured-output test for the terminal calls (Split 04 §4 live tier).

Marked ``live`` — excluded from the per-commit gate. Runs on demand:

    python -m pytest core/tests/test_build_summary_live.py -m live -v

Exercises the real structured-output API: ``build_summary`` must return a **schema-valid**
SOAP (guaranteed at the API layer, asserted anyway), with the disclaimer present, the live
``red_flags_checked`` count, and ``low_confidence_fields`` populated when the patient hedged;
``suggest_triage`` must return a band **>= the safety floor**. Scoped to the two terminal
calls (the full agent loop is covered by ``test_agent_live.py``) to bound cost. The spec pins
Opus; this environment ships an Azure GPT-5.5 key, so the terminal calls run on GPT-5.5.
"""

from __future__ import annotations

import pytest

from scribeintake.config import settings
from scribeintake.llm import build_summary_client
from scribeintake.models import SOAP, Confidence, IntakeState, SlotValue, TriageBand
from scribeintake.pricing import cost_usd
from scribeintake.safety.rules import RULES
from scribeintake.tools.build_summary import build_summary
from scribeintake.tools.suggest_triage import suggest_triage

pytestmark = pytest.mark.live

_BAND_ORDER = list(TriageBand)


@pytest.fixture(autouse=True)
def _require_azure():
    if not (settings.azure_openai_endpoint and settings.azure_openai_api_key):
        pytest.skip("Azure OpenAI credentials not configured (.env)")


def _headache_state() -> IntakeState:
    st = IntakeState(session_id="live")
    st.slots = {
        "chief_complaint": SlotValue(value="tension-type headache", confidence=Confidence.high),
        "hpi.onset": SlotValue(value="two days ago", confidence=Confidence.high),
        "hpi.character": SlotValue(value="dull, intermittent", confidence=Confidence.high),
        "hpi.severity": SlotValue(value="maybe a 5 or 6?", confidence=Confidence.medium),
        "hpi.relieving": SlotValue(value="ibuprofen helps a bit", confidence=Confidence.high),
        "medications": SlotValue(value="none", confidence=Confidence.high),
        "allergies": SlotValue(value="none reported", confidence=Confidence.high),
    }
    return st


def test_live_build_summary_and_triage():
    client = build_summary_client(settings)
    state = _headache_state()

    summary = build_summary(state, client=client, generated_at="2026-06-20T12:00:00Z")
    soap = summary.soap

    # Schema-valid (it IS a SOAP; round-trip through JSON to prove it parses).
    assert isinstance(soap, SOAP)
    SOAP.model_validate_json(soap.model_dump_json())

    # Deterministic stamping.
    assert soap.disclaimer
    assert soap.generated_at == "2026-06-20T12:00:00Z"
    assert len(soap.red_flags_checked) == len(RULES)
    # The hedged severity slot surfaces as low-confidence.
    assert "hpi.severity" in soap.subjective.low_confidence_fields
    # The model populated the subjective from the facts.
    assert soap.subjective.chief_complaint.strip()

    # Triage: band must be >= the safety floor (here self_care).
    triage = suggest_triage(state, soap, floor=TriageBand.self_care, client=client)
    assert _BAND_ORDER.index(triage.triage.band) >= _BAND_ORDER.index(TriageBand.self_care)

    # And the clamp holds even if we raise the floor above the model's pick.
    triage_hi = suggest_triage(state, soap, floor=TriageBand.gp_urgent, client=client)
    assert _BAND_ORDER.index(triage_hi.triage.band) >= _BAND_ORDER.index(TriageBand.gp_urgent)

    cost = cost_usd(
        summary.model,
        summary.usage.input_tokens + triage.usage.input_tokens,
        summary.usage.output_tokens + triage.usage.output_tokens,
    )
    print(
        f"\n[live] build_summary+triage model={summary.model} band={triage.triage.band.value} "
        f"low_conf={soap.subjective.low_confidence_fields} cost=${cost:.6f}"
    )

"""Pure mapper tests (no HTTP, no key): AssistantTurn / SOAP / trace -> DTOs with frontend names."""

from __future__ import annotations

from api import serialize
from api.tests.conftest import (
    make_agent_emergency_turn,
    make_clear_turn,
    make_completed_turn,
    make_crisis_turn,
    make_gate_emergency_turn,
    sample_soap,
)


def _json(model):
    """Serialise the way FastAPI does (by alias) so we assert on the wire field names."""
    return model.model_dump(by_alias=True)


# ----------------------------------------------------------------------------- turn / strip
def test_clear_turn_maps_message_and_strip():
    turn = make_clear_turn()
    out = _json(serialize.turn_response(turn))

    assert out["content"] == turn.assistant_text
    assert out["model"] == "gpt-5.5"
    assert out["level"] == "CLEAR"
    assert out["source"] == "gate"
    assert out["status"] == "active"
    assert out["triageFloor"] == "self_care"
    assert out["floorPinned"] is False
    assert out["readyToSummarize"] is False
    assert out["emergency"] is None

    strip = out["strip"]
    # exact frontend field names (the part that saves Split 11 a reshape)
    for key in ("ruleId", "ruleLevel", "ruleSource", "signalsView", "toolsNote", "agentNet"):
        assert key in strip
    assert strip["ruleId"] == "no rule matched"
    assert strip["ruleLevel"] == "CLEAR"
    assert strip["agentNet"] is False
    assert strip["toolsNote"] == ""
    # signals_view holds only present/true signals (dyspnea True; sbp None dropped)
    names = {s["name"] for s in strip["signalsView"]}
    assert "dyspnea" in names
    assert "sbp" not in names
    assert strip["signals"]["dyspnea"] is True


def test_clear_turn_surfaces_retrieved_sources():
    out = _json(serialize.turn_response(make_clear_turn()))
    assert out["sources"], "retrieve_guideline chunks should surface as message sources"
    src = out["sources"][0]
    assert src["chunkId"] == "chk_abc123"
    assert src["source"] == "MedlinePlus"
    assert src["url"].startswith("https://")


def test_clear_turn_trace_delta_flags_local_rows():
    out = _json(serialize.turn_response(make_clear_turn()))
    rows = {r["tool"]: r for r in out["traceDelta"]}
    assert rows["agent_step"]["local"] is False
    assert rows["record_intake"]["local"] is True
    assert rows["retrieve_guideline"]["costUsd"] == 0.0


# --------------------------------------------------------------------------------- emergency
def test_gate_emergency_serialises_template_verbatim():
    turn = make_gate_emergency_turn()
    out = _json(serialize.turn_response(turn))

    assert out["status"] == "halted"
    assert out["level"] == "EMERGENCY"
    assert out["triageFloor"] == "ER"
    assert out["floorPinned"] is True

    em = out["emergency"]
    assert em is not None
    # wording comes from the core template, not re-authored here
    assert em["heading"] == turn.template["heading"]
    assert em["body"] == turn.template["body"]
    assert em["actions"][0]["href"] == "tel:911"
    assert em["crisis"] is False
    # caption is provenance derived from the verdict
    assert "acs_chest_pain" in em["caption"]
    assert out["strip"]["toolsNote"].startswith("gate short-circuited")


def test_agent_emergency_caption_and_strip():
    out = _json(serialize.turn_response(make_agent_emergency_turn()))
    assert out["source"] == "agent"
    assert out["strip"]["agentNet"] is True
    assert "assess_escalation" in out["emergency"]["caption"]
    assert "assess_escalation" in out["strip"]["toolsNote"]


def test_crisis_turn_uses_crisis_template():
    turn = make_crisis_turn()
    out = _json(serialize.turn_response(turn))
    em = out["emergency"]
    assert em["kind"] == "crisis"
    assert em["crisis"] is True
    assert em["heading"] == turn.template["heading"]
    # crisis template offers both the lifeline and emergency actions
    hrefs = {a["href"] for a in em["actions"]}
    assert "tel:988" in hrefs
    assert "tel:911" in hrefs
    assert "Crisis template" in em["caption"]


# ----------------------------------------------------------------------------------- summary
def test_summary_maps_soap_fields():
    soap = sample_soap()
    out = _json(serialize.summary_response(soap, band="gp_routine"))

    assert out["band"] == "gp_routine"
    assert out["disclaimer"] == "Not a diagnosis. For clinician review."
    # summary fields stay snake_case per spec §3.3
    assert out["red_flags_checked"] == 21
    assert out["red_flags_triggered"] == 0
    assert out["low_confidence_fields"] == ["hpi.severity"]

    subj = {f["key"]: f for f in out["subjective"]}
    assert subj["chief_complaint"]["value"] == "dull headache"
    assert subj["medications"]["value"] == "ibuprofen as needed"
    assert subj["hpi.severity"]["low"] is True  # flagged low-confidence
    assert subj["hpi.onset"]["low"] is False

    assert out["objective"].startswith("BP 186/122 (home)")


def test_summary_observation_citation_states():
    out = _json(serialize.summary_response(sample_soap()))
    obs = out["observations"]
    cited = obs[0]
    uncited = obs[1]
    assert cited["cited"] is True and cited["uncited"] is False
    assert cited["chunk"] == "chk_bp01"
    assert cited["source"] == "MedlinePlus"
    # uncited observation has no fabricated source
    assert uncited["cited"] is False and uncited["uncited"] is True
    assert uncited["source"] == "" and uncited["chunk"] == ""


def test_completed_turn_carries_band_and_ready_flag():
    out = _json(serialize.turn_response(make_completed_turn()))
    assert out["status"] == "completed"
    assert out["readyToSummarize"] is True
    assert out["triageBand"] == "gp_routine"


# ------------------------------------------------------------------------------------- trace
def test_trace_response_totals_and_label():
    rows = [
        {
            "tool": "agent_step",
            "model": "gpt-5.5",
            "turn": 1,
            "latency_ms": 900,
            "cost_usd": 0.0023,
            "input_tokens": 1000,
            "output_tokens": 50,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 800,
        },
        {
            "tool": "retrieve_guideline",
            "model": None,
            "turn": 1,
            "latency_ms": 5,
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
        },
    ]
    out = serialize.trace_response(rows, "sess1").model_dump(by_alias=True)
    assert out["sessionId"] == "sess1"
    assert out["nTurns"] == 1
    assert out["totalCostUsd"] > 0
    assert out["pctCacheSaved"] > 0  # cache_read present -> a real saving
    assert "cache" in out["traceCostLabel"] and out["traceCostLabel"].startswith("$")
    assert {r["tool"] for r in out["rows"]} == {"agent_step", "retrieve_guideline"}

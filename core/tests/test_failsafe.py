"""Fail-safe tests — any exception in the safety path escalates, never returns CLEAR.

Spec section 18 ("never a blank failure"): if the gate itself errors, it must fail toward
caution. A silent CLEAR on a safety-path exception is the one outcome that is never allowed.
"""

import scribeintake.safety as safety
from scribeintake.models import EscalationLevel, TriageBand
from scribeintake.safety import run_gate


def _boom(*_args, **_kwargs):
    raise RuntimeError("forced safety-path failure")


def test_exception_in_evaluate_fails_safe(monkeypatch):
    monkeypatch.setattr(safety, "evaluate", _boom)
    r = run_gate("I have a mild cough")
    assert r.failed_safe is True
    assert r.verdict.level is not EscalationLevel.CLEAR
    assert r.verdict.level is EscalationLevel.URGENT
    assert "safety_check_unavailable" in r.verdict.matched_rules
    assert r.template["kind"] == "unavailable"
    assert r.floor is TriageBand.gp_urgent


def test_exception_in_extract_fails_safe(monkeypatch):
    monkeypatch.setattr(safety, "extract", _boom)
    r = run_gate("anything at all")
    assert r.failed_safe is True
    assert r.verdict.level is EscalationLevel.URGENT
    assert r.template["heading"].lower().startswith("please seek")


def test_failsafe_never_lowers_an_existing_emergency_floor(monkeypatch):
    monkeypatch.setattr(safety, "evaluate", _boom)
    r = run_gate("anything", current_floor=TriageBand.ER)
    # Fail-safe escalates to caution but must not lower a pinned ER floor.
    assert r.floor is TriageBand.ER
    assert r.failed_safe is True


def test_failsafe_message_mentions_in_person_care(monkeypatch):
    monkeypatch.setattr(safety, "evaluate", _boom)
    r = run_gate("anything")
    assert "in-person care" in r.template["body"]

"""PII redaction for shareable exports (Split 13 §3.3) — NO API key.

Asserts the documented policy in :mod:`scribeintake.redact`: direct identifiers (name, phone,
e-mail, SSN/MRN-like ids) are masked in a shareable trace/summary, while the clinical content a
reviewing clinician needs (vital values, symptom text) is preserved. Storage stays local; this
applies only to *exported* copies (§17).
"""

from __future__ import annotations

import json

from scribeintake.redact import redact_for_share, redact_text


def test_free_text_identifiers_are_masked():
    text = (
        "My name is Jane Doe, reach me at (555) 123-4567 or jane.doe@example.com. "
        "My SSN is 123-45-6789 and MRN 7421993."
    )
    out = redact_text(text)
    for leaked in ("Jane Doe", "555", "123-4567", "jane.doe@example.com", "123-45-6789", "7421993"):
        assert leaked not in out, f"identifier survived redaction: {leaked!r}"
    assert "[NAME]" in out and "[PHONE]" in out and "[EMAIL]" in out and "[ID]" in out


def test_clinical_vitals_are_preserved():
    """Vitals are clinical content, not identifiers — they must survive redaction (the policy)."""
    text = "My BP is 186/122, SpO2 is 90, heart rate 118, glucose 54 and I have bad chest pain."
    out = redact_text(text)
    for kept in ("186/122", "90", "118", "54", "bad chest pain"):
        assert kept in out, f"clinical value was wrongly masked: {kept!r}"


def test_structured_trace_redacts_identifier_keys_and_keeps_clinical():
    trace = {
        "patient_name": "John Smith",
        "phone": "555-867-5309",
        "email": "john@example.com",
        "soap": {
            "subjective": {"chief_complaint": "crushing chest pain radiating to left arm"},
            "objective": {"bp": "186/122", "spo2": 90, "hr": 118},
        },
        "rows": [{"tool": "record_intake", "note": "patient John, call 555.123.9999"}],
    }
    red = redact_for_share(trace)
    blob = json.dumps(red)

    # Identifier keys masked wholesale; free-text identifiers inside nested strings masked too.
    assert red["patient_name"] == "[REDACTED:patient_name]"
    assert red["phone"] == "[REDACTED:phone]"
    assert red["email"] == "[REDACTED:email]"
    assert "555-867-5309" not in blob and "john@example.com" not in blob
    assert "[PHONE]" in red["rows"][0]["note"]  # nested free-text phone masked

    # Clinical content preserved exactly.
    assert red["soap"]["objective"]["bp"] == "186/122"
    assert red["soap"]["objective"]["spo2"] == 90
    assert (
        red["soap"]["subjective"]["chief_complaint"]
        == "crushing chest pain radiating to left arm"
    )


def test_redaction_returns_a_copy_and_passes_clean_scalars_through():
    original = {"a": "no identifiers here", "n": 42, "ok": True, "missing": None}
    red = redact_for_share(original)
    assert red == original
    assert red is not original  # a copy, never an in-place mutation of the live data

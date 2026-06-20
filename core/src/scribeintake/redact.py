"""PII redaction for **shareable** exports (spec §17, Split 13).

The local store and the live app keep full patient data on-machine (the local-first / HIPAA
posture). This module applies **only** when a trace or summary is *exported or shared* — it
masks direct patient identifiers so a copy that leaves the machine cannot leak who the patient
is, while preserving the clinical content a reviewing clinician actually needs.

Redaction policy (documented so a reviewer is never surprised — §3.3):

**Masked (direct identifiers):**
- e-mail addresses                         → ``[EMAIL]``
- phone numbers (US-style, with/without punctuation) → ``[PHONE]``
- SSN-like ``ddd-dd-dddd``                 → ``[ID]``
- MRN / long identifier runs (≥ 6 digits)  → ``[ID]``
- names introduced by a cue ("my name is …", "I'm …", "name:") → ``[NAME]``
- values under known-identifier keys in a structured trace/summary
  (``name``/``patient_name``/``dob``/``phone``/``email``/``mrn``/``ssn``/``address``/…)
  → ``[REDACTED:<key>]``

**Kept (clinical content needed for review):**
- vital values — blood pressure (``186/122``), SpO2 (``90``), heart rate, glucose,
  temperature — these are short (≤ 3-digit) numbers, are **not** identifiers, and a shareable
  clinical trace is useless without them. (The spec lists "full BP strings" as a *candidate*
  to mask; the §3.3 "clinical content preserved" clause governs — vitals are kept, and that
  choice is asserted in ``eval/tests/test_redaction.py``.)
- all symptom text, escalation levels, rule ids, citations, costs/latency.

Everything here is pure Python (no deps, no network, no LLM) and is safe to call on a
``str``, ``dict``, or ``list`` — containers are redacted recursively.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["redact_for_share", "redact_text", "IDENTIFIER_KEYS"]

# Keys whose *value* is a direct identifier regardless of its text shape.
IDENTIFIER_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "patient_name",
        "full_name",
        "first_name",
        "last_name",
        "dob",
        "date_of_birth",
        "phone",
        "telephone",
        "phone_number",
        "email",
        "e_mail",
        "mrn",
        "ssn",
        "address",
        "street",
        "zip",
        "zipcode",
        "postcode",
        "insurance_id",
        "member_id",
    }
)

# --- text patterns (order matters: e-mail before phone before bare long-id) ---------------
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# US phone: optional +1, area code in parens or not, separators . - or space.
_PHONE = re.compile(
    r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"
)
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
# MRN / long identifier: a run of 6+ digits not adjacent to a digit or a '/' (so a BP like
# 186/122 and short vitals are never swept up). Phone/SSN are masked first.
_LONG_ID = re.compile(r"(?<![\d/])\d{6,}(?![\d/])")
# Name introduced by an explicit cue — captures 1–3 capitalized tokens.
_NAME_CUE = re.compile(
    r"\b(my name is|i am|i'm|this is|name:|patient:)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
    re.IGNORECASE,
)


def redact_text(text: str) -> str:
    """Mask direct identifiers in a free-text string (see module policy)."""
    if not text:
        return text
    out = _EMAIL.sub("[EMAIL]", text)
    out = _PHONE.sub("[PHONE]", out)
    out = _SSN.sub("[ID]", out)
    out = _LONG_ID.sub("[ID]", out)
    # Keep the cue word, mask only the captured name (group 2).
    out = _NAME_CUE.sub(lambda m: f"{m.group(1)} [NAME]", out)
    return out


def redact_for_share(value: Any) -> Any:
    """Return a redacted **copy** of ``value`` safe to export/share.

    Strings are masked per :func:`redact_text`. Dicts/lists are redacted recursively; a dict
    value under an :data:`IDENTIFIER_KEYS` key is replaced wholesale with ``[REDACTED:<key>]``
    (its shape doesn't matter — the key tells us it's an identifier). Non-string scalars
    (ints/floats/bools/None) pass through unchanged — clinical numbers are not identifiers.
    """
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, val in value.items():
            if isinstance(key, str) and key.lower() in IDENTIFIER_KEYS:
                out[key] = f"[REDACTED:{key.lower()}]"
            else:
                out[key] = redact_for_share(val)
        return out
    if isinstance(value, list):
        return [redact_for_share(v) for v in value]
    return value

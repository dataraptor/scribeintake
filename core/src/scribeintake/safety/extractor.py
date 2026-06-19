"""Deterministic signal extractor — the layer that feeds the safety gate.

**This module contains no LLM call, no network, and no I/O.** It is a pure function:
the same raw text + prior signals always produce the same :class:`Signals`. That purity
is *why* the must-escalate set can be gated at 100% (spec section 10) — the guarantee is
``assert``, not "the model usually catches it".

Two reproducible sources feed the gate (spec section 10 / Appendix A):

1. **Raw-text pattern match** on the current message — curated regex/keyword lists for
   explicit danger phrases plus number parsing for vitals.
2. **Structured signals already in ``intake_state``** — passed in as ``prior`` and merged.

Extraction is **clause-aware negated** (mirrors the ``.dc.html`` mockup ``extract()``):
a signal phrase sitting inside a negated clause (``no chest pain``, ``denies headache``,
``don't have neck stiffness``) does **not** set the signal. Booleans merge with ``prior``
by OR (a danger seen earlier stays seen — consistent with monotonic escalation); numerics
take the latest non-null reading.

**Scope: adult patients (v1).** Pediatric/neonatal red flags are deliberately out of
scope (spec Appendix A) and not extracted here.
"""

from __future__ import annotations

import re

from ..models import Signals

# Clause boundary characters: a negation only suppresses a phrase in its own clause.
_BOUNDARY = set(".!?;\n—")

# A clause counts as negated if it contains one of these tokens before/within the phrase.
_NEGATION = re.compile(r"\b(no|not|without|deny|denies|negative|never|none)\b|n't")

# --------------------------------------------------------------------------------------
# Symptom phrase patterns (boolean signals). Conservative/broad by design (spec section
# 10, mitigation 2): biased toward flagging — false alarms are measured in Split 06/07.
# Every phrase the mockup ``extract()`` covers is reproduced here and extended with
# paraphrases. Patterns are matched case-insensitively against the lowered message.
# --------------------------------------------------------------------------------------
_SYMPTOM_PATTERNS: dict[str, str] = {
    "chest_pain": (
        r"chest (pain|tight|pressure|heav|discomfort|ache|hurt|sore)"
        r"|tightness in (my |the )?chest"
        r"|pain in (my |the )?chest"
        r"|(crushing|squeezing) (chest|in my chest)"
    ),
    "pain_radiation_arm_jaw_back": (
        r"(left |right )?arm (hurt|pain|ache|numb|weak)"
        r"|spread(s|ing)? to (my )?(left |right )?(arm|jaw|back|shoulder)"
        r"|radiat"
        r"|into my (left |right )?(arm|jaw|back)"
        r"|(arm|jaw|back|shoulder) pain"
    ),
    "diaphoresis": r"sweat|diaphor|clammy",
    "dyspnea": (
        r"short(ness)? of breath|breathless|hard to breathe|out of breath"
        r"|can'?t catch my breath|trouble breathing|difficulty breathing|winded"
    ),
    "nausea": r"nause|queasy|sick to my stomach",
    "face_droop": r"face (is |feels )?(droop|drooping|sagging)|droopy|one side of my face",
    "limb_weakness": (
        r"arm (went |feels |is )?weak|leg (went |feels |is )?weak"
        r"|weak(ness)? on one side|one side.*weak"
    ),
    "speech_difficulty": r"slur|can'?t speak|trouble speaking|can'?t get my words",
    "sudden_vision_loss": (
        r"lost (my )?vision|can'?t see|vision (went|loss)|blurry vision|sudden.*vision"
    ),
    "sudden_confusion": r"confus|disorient|can'?t think straight",
    "worst_headache_ever": r"worst headache|worst.*head.*(life|ever)",
    "thunderclap_headache": (
        r"thunderclap|came on suddenly.*headache"
        r"|sudden.{0,12}(severe|terrible).{0,12}headache|head.*explod"
    ),
    "neck_stiffness": r"neck (is )?stiff|stiff neck",
    "fever": r"fever|burning up|febrile",
    # Airway compromise: "throat closing up" / "choking" / "can't get air" are mapped to
    # cant_breathe (not just throat_or_tongue_swelling) so that airway emergencies escalate
    # via `respiratory_distress` even when no allergen is stated. See the safety note below.
    "cant_breathe": (
        r"can'?t breathe|cannot breathe|choking|can'?t get air"
        r"|throat.*clos|closing up"
    ),
    "throat_or_tongue_swelling": r"throat.*swell|tongue.*swell|closing up|throat.*clos",
    "hives": r"hives|welts|breaking out in",
    "known_allergen_exposure": r"allerg|peanut|bee sting|shellfish|nuts|penicillin",
    "suicidal_ideation": (
        r"suicid|kill myself|end (my life|it all)"
        r"|don'?t want to (live|be here)|not worth living|better off dead"
    ),
    "self_harm_intent": r"hurt myself|harm myself|cut(ting)? myself",
    "vaginal_bleeding": r"vaginal bleed|bleeding.*(vagina|down there)",
    "severe_abdominal_pain": (
        r"severe (abdominal|stomach|belly) pain"
        r"|(abdominal|stomach|belly) pain.*severe"
        r"|worst (abdominal|stomach|belly) pain"
    ),
    "vomiting_blood": r"vomit.*blood|throwing up blood|coughing up blood",
    "rigid_abdomen": (
        r"rigid (abdomen|belly|stomach)|(abdomen|belly|stomach) (is )?(rigid|hard|board)"
        r"|board.?like"
    ),
    "head_injury": r"hit my head|head injury|banged my head|fell.*head",
    "on_anticoagulant": r"blood thinner|warfarin|eliquis|xarelto|apixaban|coumadin",
    "pregnant": r"pregnan|expecting|weeks along",
}

# Compile once (module import). Pure regex — no I/O.
_COMPILED: dict[str, re.Pattern[str]] = {
    name: re.compile(pat) for name, pat in _SYMPTOM_PATTERNS.items()
}

# --------------------------------------------------------------------------------------
# Number parsing. Each (field, pattern, low, high) parses a vital and range-guards it so
# stray digits ("I'm 45 years old") don't masquerade as a reading.
# --------------------------------------------------------------------------------------
_BP_RE = re.compile(r"(\d{2,3})\s*(?:/|over)\s*(\d{2,3})")
_GLUCOSE_RE = re.compile(r"(?:glucose|sugar|\bbg\b|blood sugar)\D{0,8}(\d{2,4})")
_SPO2_RE = re.compile(r"(?:o2|sat|spo2|oxygen)\D{0,8}(\d{2,3})")
_HR_RE = re.compile(r"(?:heart rate|pulse|\bhr\b)\D{0,8}(\d{2,3})")
_TEMP_RE = re.compile(r"(?:temp(?:erature)?|fever (?:of|is|at))\D{0,8}(\d{2,3}(?:\.\d)?)")

# Fever threshold (°F). At/above this a parsed temperature also sets the `fever` boolean —
# conservative, since many rules key on `fever` rather than the raw number.
_FEVER_F = 100.4


def _clause_negated(text: str, start: int) -> bool:
    """True if a negation precedes the phrase at ``start`` within the same clause.

    Walks back from ``start`` to the nearest clause boundary, then checks only the text
    **before** the matched phrase for a negation token. Checking before-only (not the
    phrase itself) is the key fix over the mockup: it keeps "no chest pain" / "denies
    headache" negated while letting danger phrases that contain a contraction — "can't
    breathe", "can't catch my breath" — flag correctly (the "n't" is part of the symptom,
    not a negation of it).
    """
    cs = 0
    for i in range(start, -1, -1):
        if text[i] in _BOUNDARY:
            cs = i + 1
            break
    before = text[cs:start]
    return _NEGATION.search(before) is not None


def _phrase_present(text: str, pattern: re.Pattern[str]) -> bool:
    """True if any match of ``pattern`` occurs in a non-negated clause.

    Iterating *all* matches (not just the first) means "no pain earlier, but I have chest
    pain now" still flags — strictly safer than the mockup's first-match-only check.
    """
    for m in pattern.finditer(text):
        if not _clause_negated(text, m.start()):
            return True
    return False


def _parse_int(value: str, low: int, high: int) -> int | None:
    """Parse to int and range-guard; return ``None`` if out of plausible range."""
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return None
    return n if low <= n <= high else None


def extract(text: str, prior: Signals | None = None) -> Signals:
    """Extract :class:`Signals` from raw ``text`` merged with ``prior`` state.

    Pure and deterministic: no LLM, no network, no randomness. Booleans OR with ``prior``
    (monotonic — a flag once seen stays seen); numerics take the latest non-null reading.

    Args:
        text: The current patient message (raw).
        prior: Signals already accumulated this session (typically from ``intake_state``).

    Returns:
        A new :class:`Signals` (the gate's only input).
    """
    base = prior.model_dump() if prior is not None else Signals().model_dump()
    t = (text or "").lower()

    # Booleans: OR-merge — only ever set True (never clear a prior True).
    for name, pattern in _COMPILED.items():
        if not base.get(name) and _phrase_present(t, pattern):
            base[name] = True

    # Numerics: latest non-null wins; keep prior when nothing new parses.
    bp = _BP_RE.search(t)
    if bp:
        sbp = _parse_int(bp.group(1), 50, 300)
        dbp = _parse_int(bp.group(2), 30, 200)
        if sbp is not None:
            base["sbp"] = sbp
        if dbp is not None:
            base["dbp"] = dbp

    gl = _GLUCOSE_RE.search(t)
    if gl:
        glucose = _parse_int(gl.group(1), 10, 1500)
        if glucose is not None:
            base["glucose_mgdl"] = glucose

    o2 = _SPO2_RE.search(t)
    if o2:
        spo2 = _parse_int(o2.group(1), 50, 100)
        if spo2 is not None:
            base["spo2"] = spo2

    hr = _HR_RE.search(t)
    if hr:
        heart = _parse_int(hr.group(1), 30, 250)
        if heart is not None:
            base["hr"] = heart

    temp = _TEMP_RE.search(t)
    if temp:
        tf = _parse_int(temp.group(1), 90, 115)
        if tf is not None:
            base["temp_f"] = tf
            if float(temp.group(1)) >= _FEVER_F:
                base["fever"] = True

    return Signals(**base)

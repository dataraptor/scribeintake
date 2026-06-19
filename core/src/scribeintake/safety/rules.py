"""Declarative red-flag rule engine (spec section 10, Appendix A) — **no LLM**.

The rules are pure predicates over :class:`Signals` (the extractor's output, never raw
model text). Pure predicates are trivially unit-testable — ``assert evaluate(s).level ==
EMERGENCY`` — which is what lets the must-escalate set be gated at 100% in CI with no API
key. The two ``…_emergency`` escalations from Appendix A are **separate, single-level
rows** (not branches inside one predicate) so each is independently asserted.

**Scope: adult patients (v1)** — pediatric/neonatal red flags are out of scope (Appendix A).

Deviation from Appendix A (deliberate, safety-first): one extra EMERGENCY rule,
``airway_swelling`` (``throat_or_tongue_swelling`` on its own → EMERGENCY). Appendix A's
``anaphylaxis`` rule is allergen-conditioned, so airway swelling with *no stated allergen*
would fall through to CLEAR — a dangerous miss for a safety-first tool (split doc section
8). ``anaphylaxis`` is kept as-is (it still covers the ``hives + dyspnea + allergen``
branch); ``airway_swelling`` closes the bare-airway gap. The extractor also maps "throat
closing up" / "choking" to ``cant_breathe`` as a second, independent catch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import RULES_VERSION
from ..models import EscalationLevel, EscalationSource, SafetyVerdict, Signals, TriageBand

# Highest-severity-wins ordering for the verdict level.
_LEVEL_RANK: dict[EscalationLevel, int] = {
    EscalationLevel.CLEAR: 0,
    EscalationLevel.URGENT: 1,
    EscalationLevel.EMERGENCY: 2,
}

# Monotonic triage-floor ordering (a floor, once set, never lowers).
_BAND_ORDER: list[TriageBand] = [
    TriageBand.self_care,
    TriageBand.gp_routine,
    TriageBand.gp_urgent,
    TriageBand.ER,
]

# Each escalation level maps to the floor it pins.
_LEVEL_TO_BAND: dict[EscalationLevel, TriageBand | None] = {
    EscalationLevel.CLEAR: None,
    EscalationLevel.URGENT: TriageBand.gp_urgent,
    EscalationLevel.EMERGENCY: TriageBand.ER,
}


# --------------------------------------------------------------------------------------
# Shared sub-predicates (kept tiny and pure; reused across rows to stay DRY).
# --------------------------------------------------------------------------------------
def _acs_features(s: Signals) -> bool:
    """ACS-associated features for chest pain."""
    return s.pain_radiation_arm_jaw_back or s.diaphoresis or s.dyspnea or s.nausea


def _high_bp(s: Signals) -> bool:
    return (s.sbp is not None and s.sbp >= 180) or (s.dbp is not None and s.dbp >= 120)


def _end_organ(s: Signals) -> bool:
    """End-organ symptoms that turn a hypertensive *urgency* into an *emergency*."""
    return (
        s.chest_pain
        or s.dyspnea
        or s.sudden_vision_loss
        or s.sudden_confusion
        or s.worst_headache_ever
    )


# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    """One declarative red-flag rule (spec section 10).

    ``condition`` is a pure predicate over :class:`Signals`; ``level`` is single-valued
    (escalations are separate rows). ``message_key`` selects a deterministic template.
    """

    id: str
    condition: Callable[[Signals], bool]
    level: EscalationLevel
    message_key: str
    source_citation: str
    crisis: bool = False
    rules_version: str = field(default=RULES_VERSION)


# The rule set (Appendix A table + the deliberate `airway_swelling` row). EMERGENCY rows
# listed first for readable display; `evaluate` collects all matches and takes the max, so
# order does not affect correctness.
RULES: list[Rule] = [
    # ---- EMERGENCY ----
    Rule(
        "acs_chest_pain",
        lambda s: s.chest_pain and _acs_features(s),
        EscalationLevel.EMERGENCY,
        "emergency",
        "MedlinePlus Heart Attack; CDC Heart Attack Symptoms",
    ),
    Rule(
        "stroke_fast",
        lambda s: (
            s.face_droop
            or s.limb_weakness
            or s.speech_difficulty
            or s.sudden_vision_loss
            or s.sudden_confusion
        ),
        EscalationLevel.EMERGENCY,
        "emergency",
        "CDC Stroke (FAST); MedlinePlus Stroke",
    ),
    Rule(
        "thunderclap_headache",
        lambda s: s.worst_headache_ever or s.thunderclap_headache,
        EscalationLevel.EMERGENCY,
        "emergency",
        "MedlinePlus Headache / when to seek care",
    ),
    Rule(
        "meningitis_signs",
        lambda s: s.fever and s.neck_stiffness,
        EscalationLevel.EMERGENCY,
        "emergency",
        "MedlinePlus / CDC Meningitis",
    ),
    Rule(
        "suicidal_crisis",
        lambda s: s.suicidal_ideation or s.self_harm_intent,
        EscalationLevel.EMERGENCY,
        "crisis",
        "NIMH; 988 Suicide & Crisis Lifeline",
        crisis=True,
    ),
    Rule(
        "respiratory_distress",
        lambda s: s.cant_breathe or (s.dyspnea and s.spo2 is not None and s.spo2 < 92),
        EscalationLevel.EMERGENCY,
        "emergency",
        "MedlinePlus Breathing Problems",
    ),
    Rule(
        "anaphylaxis",
        lambda s: (
            (s.throat_or_tongue_swelling or (s.hives and s.dyspnea)) and s.known_allergen_exposure
        ),
        EscalationLevel.EMERGENCY,
        "emergency",
        "MedlinePlus Anaphylaxis",
    ),
    Rule(
        # Deliberate safety addition (see module docstring): airway swelling alone escalates.
        "airway_swelling",
        lambda s: s.throat_or_tongue_swelling,
        EscalationLevel.EMERGENCY,
        "emergency",
        "MedlinePlus Anaphylaxis / Breathing Problems",
    ),
    Rule(
        "htn_emergency",
        lambda s: _high_bp(s) and _end_organ(s),
        EscalationLevel.EMERGENCY,
        "emergency",
        "NHLBI / MedlinePlus High Blood Pressure",
    ),
    Rule(
        "severe_hypoglycemia",
        lambda s: s.glucose_mgdl is not None and s.glucose_mgdl < 54,
        EscalationLevel.EMERGENCY,
        "emergency",
        "NIDDK Low Blood Glucose; MedlinePlus",
    ),
    Rule(
        "hyperglycemia_dka",
        lambda s: (
            s.glucose_mgdl is not None
            and s.glucose_mgdl > 300
            and (s.nausea or s.sudden_confusion or s.dyspnea)
        ),
        EscalationLevel.EMERGENCY,
        "emergency",
        "NIDDK DKA / MedlinePlus",
    ),
    Rule(
        "anticoag_head_injury_emergency",
        lambda s: s.on_anticoagulant and s.head_injury and (s.sudden_confusion or s.limb_weakness),
        EscalationLevel.EMERGENCY,
        "emergency",
        "MedlinePlus Head Injury",
    ),
    Rule(
        "pregnancy_bleeding",
        lambda s: s.pregnant and (s.vaginal_bleeding or s.severe_abdominal_pain),
        EscalationLevel.EMERGENCY,
        "emergency",
        "MedlinePlus Bleeding during pregnancy",
    ),
    Rule(
        "gi_bleed",
        lambda s: s.vomiting_blood or (s.severe_abdominal_pain and s.rigid_abdomen),
        EscalationLevel.EMERGENCY,
        "emergency",
        "MedlinePlus Gastrointestinal Bleeding",
    ),
    Rule(
        "sepsis_signs",
        lambda s: (
            s.fever and (s.sudden_confusion or (s.hr is not None and s.hr > 110 and s.dyspnea))
        ),
        EscalationLevel.EMERGENCY,
        "emergency",
        "CDC About Sepsis",
    ),
    Rule(
        "hypoxia_emergency",
        lambda s: s.spo2 is not None and s.spo2 < 92 and s.cant_breathe,
        EscalationLevel.EMERGENCY,
        "emergency",
        "MedlinePlus Hypoxemia",
    ),
    # ---- URGENT ----
    Rule(
        "chest_pain_isolated",
        lambda s: s.chest_pain and not _acs_features(s),
        EscalationLevel.URGENT,
        "urgent",
        "MedlinePlus Chest Pain",
    ),
    Rule(
        "htn_urgency",
        lambda s: _high_bp(s) and not _end_organ(s),
        EscalationLevel.URGENT,
        "urgent",
        "NHLBI / MedlinePlus High Blood Pressure",
    ),
    Rule(
        "anticoag_head_injury",
        lambda s: s.on_anticoagulant and s.head_injury,
        EscalationLevel.URGENT,
        "urgent",
        "MedlinePlus Head Injury",
    ),
    Rule(
        "hypoxia",
        lambda s: s.spo2 is not None and s.spo2 < 92,
        EscalationLevel.URGENT,
        "urgent",
        "MedlinePlus Hypoxemia",
    ),
    Rule(
        "severe_abdominal_isolated",
        lambda s: (
            s.severe_abdominal_pain and not (s.vomiting_blood or s.rigid_abdomen or s.pregnant)
        ),
        EscalationLevel.URGENT,
        "urgent",
        "MedlinePlus Abdominal Pain",
    ),
]


def evaluate(signals: Signals) -> SafetyVerdict:
    """Evaluate **all** rules; the verdict is the highest matched level (deterministic).

    Collects every matching rule id (stable list order). ``crisis`` is True if any matched
    rule is a crisis rule. ``source`` is always ``gate`` — this is the deterministic net.
    No rule match → ``CLEAR`` with an empty ``matched_rules``.
    """
    matched = [r for r in RULES if r.condition(signals)]
    if not matched:
        return SafetyVerdict(
            level=EscalationLevel.CLEAR,
            matched_rules=[],
            source=EscalationSource.gate,
            rules_version=RULES_VERSION,
            crisis=False,
        )
    level = max((r.level for r in matched), key=lambda lvl: _LEVEL_RANK[lvl])
    return SafetyVerdict(
        level=level,
        matched_rules=[r.id for r in matched],
        source=EscalationSource.gate,
        rules_version=RULES_VERSION,
        crisis=any(r.crisis for r in matched),
    )


def raise_floor(current: TriageBand, level: EscalationLevel) -> TriageBand:
    """Return the higher of ``current`` and the band ``level`` pins (never lower).

    This is the monotonic-escalation guarantee: ``EMERGENCY`` pins ``ER``, ``URGENT`` pins
    ``gp_urgent``, ``CLEAR`` pins nothing. A floor once set is never lowered.
    """
    target = _LEVEL_TO_BAND.get(level)
    if target is None:
        return current
    return max(current, target, key=_BAND_ORDER.index)

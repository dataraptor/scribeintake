"""Exhaustive contract tests for the rule engine — the frozen must-escalate gate.

This file IS the per-commit reliability contract (spec section 15): every EMERGENCY rule
row has a positive ``assert evaluate(signals).level == EMERGENCY`` with zero API calls.
The ``CANONICAL`` table is asserted to cover **every** rule, so coverage cannot silently
regress when a rule is added.
"""

import pytest

from scribeintake.models import EscalationLevel, Signals, TriageBand
from scribeintake.safety.rules import RULES, evaluate, raise_floor

# A canonical signal set per rule id: the minimal flags that make THAT rule fire at its
# declared level. EMERGENCY sets may co-fire other EMERGENCY rules (e.g. a FAST sign), but
# their resolved level is still EMERGENCY; URGENT sets are isolated to resolve to URGENT.
CANONICAL: dict[str, dict] = {
    # ---- EMERGENCY ----
    "acs_chest_pain": dict(chest_pain=True, diaphoresis=True),
    "stroke_fast": dict(face_droop=True),
    "thunderclap_headache": dict(worst_headache_ever=True),
    "meningitis_signs": dict(fever=True, neck_stiffness=True),
    "suicidal_crisis": dict(suicidal_ideation=True),
    "respiratory_distress": dict(cant_breathe=True),
    "anaphylaxis": dict(hives=True, dyspnea=True, known_allergen_exposure=True),
    "airway_swelling": dict(throat_or_tongue_swelling=True),
    "htn_emergency": dict(sbp=185, dyspnea=True),
    "severe_hypoglycemia": dict(glucose_mgdl=40),
    "hyperglycemia_dka": dict(glucose_mgdl=450, nausea=True),
    "anticoag_head_injury_emergency": dict(
        on_anticoagulant=True, head_injury=True, limb_weakness=True
    ),
    "pregnancy_bleeding": dict(pregnant=True, vaginal_bleeding=True),
    "gi_bleed": dict(vomiting_blood=True),
    "sepsis_signs": dict(fever=True, hr=120, dyspnea=True),
    "hypoxia_emergency": dict(spo2=85, cant_breathe=True),
    # ---- URGENT ----
    "chest_pain_isolated": dict(chest_pain=True),
    "htn_urgency": dict(sbp=185),
    "anticoag_head_injury": dict(on_anticoagulant=True, head_injury=True),
    "hypoxia": dict(spo2=88),
    "severe_abdominal_isolated": dict(severe_abdominal_pain=True),
}

_RULE_BY_ID = {r.id: r for r in RULES}
EMERGENCY_IDS = [r.id for r in RULES if r.level is EscalationLevel.EMERGENCY]
URGENT_IDS = [r.id for r in RULES if r.level is EscalationLevel.URGENT]


def test_rule_set_shape():
    """20 Appendix-A rows + 1 deliberate safety addition (airway_swelling) = 21 rules."""
    assert len(RULES) == 21
    assert len(EMERGENCY_IDS) == 16
    assert len(URGENT_IDS) == 5
    # Rule ids are unique.
    assert len({r.id for r in RULES}) == len(RULES)


def test_canonical_covers_every_rule():
    """Coverage cannot silently regress: every rule has a canonical positive set."""
    assert set(CANONICAL) == {r.id for r in RULES}


@pytest.mark.parametrize("rule_id", list(CANONICAL))
def test_each_rule_fires_at_its_level(rule_id):
    """Positive test per rule row: it fires, and the verdict resolves to its level."""
    rule = _RULE_BY_ID[rule_id]
    verdict = evaluate(Signals(**CANONICAL[rule_id]))
    assert rule_id in verdict.matched_rules
    assert verdict.level is rule.level


@pytest.mark.parametrize("rule_id", EMERGENCY_IDS)
def test_every_emergency_row_escalates(rule_id):
    """The frozen must-escalate contract: 100% of EMERGENCY rows assert EMERGENCY."""
    verdict = evaluate(Signals(**CANONICAL[rule_id]))
    assert verdict.level is EscalationLevel.EMERGENCY


# --------------------------------------------------------------------------- negatives
def test_benign_signals_are_clear():
    verdict = evaluate(Signals())
    assert verdict.level is EscalationLevel.CLEAR
    assert verdict.matched_rules == []
    assert verdict.crisis is False


def test_chest_pain_alone_is_urgent_not_emergency():
    """Near-miss for acs_chest_pain: isolated chest pain is URGENT, never EMERGENCY."""
    verdict = evaluate(Signals(chest_pain=True))
    assert verdict.level is EscalationLevel.URGENT
    assert "chest_pain_isolated" in verdict.matched_rules
    assert "acs_chest_pain" not in verdict.matched_rules


def test_high_bp_without_end_organ_is_urgent_not_emergency():
    """Near-miss for htn_emergency: BP 185 with no end-organ symptom is URGENT."""
    verdict = evaluate(Signals(sbp=185))
    assert verdict.level is EscalationLevel.URGENT
    assert "htn_urgency" in verdict.matched_rules
    assert "htn_emergency" not in verdict.matched_rules


def test_mildly_high_glucose_without_symptoms_is_clear():
    """Near-miss for DKA: glucose 350 with no nausea/confusion/dyspnea does not fire."""
    verdict = evaluate(Signals(glucose_mgdl=350))
    assert verdict.level is EscalationLevel.CLEAR


def test_normal_spo2_is_clear():
    assert evaluate(Signals(spo2=97)).level is EscalationLevel.CLEAR


def test_anaphylaxis_requires_allergen_but_airway_still_escalates():
    """Documented safety decision (split doc section 8): the Appendix-A anaphylaxis rule is
    allergen-conditioned, so hives+dyspnea with NO allergen does not fire it. Bare airway
    swelling, however, escalates on its own via airway_swelling.
    """
    # hives + dyspnea, no allergen -> anaphylaxis does NOT fire (and nothing else does).
    no_allergen = evaluate(Signals(hives=True, dyspnea=True))
    assert "anaphylaxis" not in no_allergen.matched_rules
    assert no_allergen.level is EscalationLevel.CLEAR
    # bare throat/tongue swelling, no allergen -> airway_swelling escalates anyway.
    airway = evaluate(Signals(throat_or_tongue_swelling=True))
    assert "airway_swelling" in airway.matched_rules
    assert airway.level is EscalationLevel.EMERGENCY


def test_anticoag_head_injury_needs_both():
    assert evaluate(Signals(on_anticoagulant=True)).level is EscalationLevel.CLEAR
    assert evaluate(Signals(head_injury=True)).level is EscalationLevel.CLEAR


# --------------------------------------------------------------- max-level & crisis
def test_max_level_selection_lists_both_rules():
    """A set matching both an URGENT and an EMERGENCY rule resolves to EMERGENCY."""
    verdict = evaluate(Signals(spo2=85, cant_breathe=True))
    assert verdict.level is EscalationLevel.EMERGENCY
    assert "hypoxia" in verdict.matched_rules  # URGENT row
    assert "hypoxia_emergency" in verdict.matched_rules  # EMERGENCY row


def test_crisis_flag_set_only_for_crisis_rule():
    assert evaluate(Signals(suicidal_ideation=True)).crisis is True
    assert evaluate(Signals(self_harm_intent=True)).crisis is True
    # A non-crisis EMERGENCY does not set the crisis flag.
    assert evaluate(Signals(cant_breathe=True)).crisis is False


# ------------------------------------------------------------------ purity / source
def test_evaluate_is_pure_and_deterministic():
    signals = Signals(chest_pain=True, diaphoresis=True, sbp=185)
    first = evaluate(signals)
    second = evaluate(signals)
    assert first.model_dump() == second.model_dump()
    # evaluate must not mutate its input.
    assert signals == Signals(chest_pain=True, diaphoresis=True, sbp=185)


def test_verdict_source_is_always_gate():
    assert evaluate(Signals(cant_breathe=True)).source.value == "gate"
    assert evaluate(Signals()).source.value == "gate"


# --------------------------------------------------------------------- raise_floor
def test_raise_floor_maps_levels():
    assert raise_floor(TriageBand.self_care, EscalationLevel.EMERGENCY) is TriageBand.ER
    assert raise_floor(TriageBand.self_care, EscalationLevel.URGENT) is TriageBand.gp_urgent
    assert raise_floor(TriageBand.self_care, EscalationLevel.CLEAR) is TriageBand.self_care

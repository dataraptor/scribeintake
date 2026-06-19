"""Tests for the deterministic signal extractor (no LLM, no I/O).

Covers: each signal phrase (+ paraphrases) → its boolean; clause-aware negation; mixed
clauses; number parsing; benign sentences; and the prior-merge (OR booleans, latest
numerics) behavior.
"""

import pytest

from scribeintake.models import Signals
from scribeintake.safety.extractor import extract

# (message, signal_name) pairs — each phrase must set its boolean True.
POSITIVE_PHRASES = [
    ("I have crushing chest pain", "chest_pain"),
    ("there's tightness in my chest", "chest_pain"),
    ("my chest hurts", "chest_pain"),
    ("the pain spreads to my left arm", "pain_radiation_arm_jaw_back"),
    ("my left arm hurts", "pain_radiation_arm_jaw_back"),
    ("it radiates to my jaw", "pain_radiation_arm_jaw_back"),
    ("I'm sweating a lot", "diaphoresis"),
    ("I feel clammy", "diaphoresis"),
    ("I'm short of breath", "dyspnea"),
    ("it's hard to breathe", "dyspnea"),
    ("I feel nauseous", "nausea"),
    ("I'm queasy", "nausea"),
    ("one side of my face is drooping", "face_droop"),
    ("my face feels droopy", "face_droop"),
    ("my arm went weak", "limb_weakness"),
    ("my speech is slurred", "speech_difficulty"),
    ("I suddenly lost my vision", "sudden_vision_loss"),
    ("I feel confused and disoriented", "sudden_confusion"),
    ("this is the worst headache of my life", "worst_headache_ever"),
    ("it was a thunderclap headache", "thunderclap_headache"),
    ("my neck is stiff", "neck_stiffness"),
    ("I have a fever", "fever"),
    ("I can't breathe", "cant_breathe"),
    ("my throat is closing up", "cant_breathe"),
    ("my tongue is swelling", "throat_or_tongue_swelling"),
    ("I'm breaking out in hives", "hives"),
    ("I'm allergic to peanuts", "known_allergen_exposure"),
    ("I want to kill myself", "suicidal_ideation"),
    ("I've been cutting myself", "self_harm_intent"),
    ("I have vaginal bleeding", "vaginal_bleeding"),
    ("severe stomach pain", "severe_abdominal_pain"),
    ("I'm vomiting blood", "vomiting_blood"),
    ("my belly is rigid and hard", "rigid_abdomen"),
    ("I hit my head", "head_injury"),
    ("I take a blood thinner", "on_anticoagulant"),
    ("I'm pregnant", "pregnant"),
]


@pytest.mark.parametrize("message,signal", POSITIVE_PHRASES)
def test_phrase_sets_signal(message, signal):
    assert getattr(extract(message), signal) is True


# ------------------------------------------------------------------- negation
NEGATED = [
    ("I have no chest pain", "chest_pain"),
    ("no sweating", "diaphoresis"),
    ("no trouble breathing", "dyspnea"),
    ("denies the worst headache of his life", "worst_headache_ever"),
    ("I don't have neck stiffness", "neck_stiffness"),
    ("without any fever", "fever"),
    ("never had chest pain", "chest_pain"),
]


@pytest.mark.parametrize("message,signal", NEGATED)
def test_negated_phrase_does_not_set_signal(message, signal):
    assert getattr(extract(message), signal) is False


def test_compound_negation_two_signals():
    """'no sweating or trouble breathing' negates both diaphoresis and dyspnea."""
    s = extract("no sweating or trouble breathing")
    assert s.diaphoresis is False
    assert s.dyspnea is False


def test_mixed_clause_negation():
    """'my chest hurts but no sweating' -> chest_pain True, diaphoresis False."""
    s = extract("my chest hurts but no sweating")
    assert s.chest_pain is True
    assert s.diaphoresis is False


def test_negation_is_clause_local():
    """A negation in one clause must not suppress a signal in a later clause."""
    s = extract("no fever. I have crushing chest pain")
    assert s.fever is False
    assert s.chest_pain is True


# --------------------------------------------------------------- number parsing
def test_bp_slash_and_over():
    assert (extract("my BP is 190/130").sbp, extract("my BP is 190/130").dbp) == (190, 130)
    s = extract("home monitor said 186 over 122")
    assert (s.sbp, s.dbp) == (186, 122)


def test_glucose_spo2_hr_temp():
    assert extract("my glucose was 45").glucose_mgdl == 45
    assert extract("O2 sat 88").spo2 == 88
    assert extract("my pulse is 120").hr == 120
    s = extract("my temperature is 103")
    assert s.temp_f == 103


def test_high_temperature_also_sets_fever():
    s = extract("my temperature is 102")
    assert s.temp_f == 102
    assert s.fever is True


def test_out_of_range_numbers_are_ignored():
    # "45 years old" must not become a glucose/spo2 reading; no vital keyword nearby.
    s = extract("I'm 45 years old and feel fine")
    assert s.glucose_mgdl is None
    assert s.spo2 is None
    # spo2 range guard: an oxygen reading of 5 is implausible -> ignored.
    assert extract("oxygen 5").spo2 is None


def test_benign_sentence_sets_nothing():
    s = extract("I've had a mild runny nose for a couple of days")
    assert s == Signals()


# ------------------------------------------------------------------- prior merge
def test_prior_booleans_or_merge():
    prior = Signals(chest_pain=True)
    # New message doesn't mention chest pain, but the prior flag persists (monotonic).
    s = extract("I'm also sweating now", prior)
    assert s.chest_pain is True
    assert s.diaphoresis is True


def test_prior_numeric_kept_when_absent_and_replaced_when_present():
    prior = Signals(sbp=150, dbp=95)
    kept = extract("I feel a bit dizzy", prior)
    assert (kept.sbp, kept.dbp) == (150, 95)
    replaced = extract("now it reads 190/120", prior)
    assert (replaced.sbp, replaced.dbp) == (190, 120)


def test_extract_is_pure():
    prior = Signals(chest_pain=True)
    out = extract("I'm sweating", prior)
    # The prior object must not be mutated.
    assert prior == Signals(chest_pain=True)
    assert out is not prior

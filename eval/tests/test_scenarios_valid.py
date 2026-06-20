"""Schema validity + target-mix + rule-coverage invariants over the gold set (no API key).

Complements ``test_must_escalate_gate.py`` (which checks the *labels* against the code gate):
this module checks the *dataset shape* — that every YAML loads, the §15 target mix is met, a
held-out slice exists, every EMERGENCY rule is exercised, gold_soap presence is honest
(urgent/routine have one; emergencies don't), and ids match filenames.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from eval.scenario import Scenario, ScenarioCategory, load_scenarios
from scribeintake.models import EscalationLevel, EscalationSource
from scribeintake.safety.rules import RULES

from .test_must_escalate_gate import gate_trace

SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "scenarios"

ALL_SCENARIOS = load_scenarios(SCENARIOS_DIR)

# The EMERGENCY rule ids are the behavioural contract the frozen set must exercise (Split 02).
EMERGENCY_RULE_IDS = {r.id for r in RULES if r.level is EscalationLevel.EMERGENCY}


def test_all_scenarios_load_without_error():
    """Loading the whole directory raises on nothing and yields a non-empty set."""
    assert ALL_SCENARIOS, "no scenarios loaded"
    assert all(isinstance(s, Scenario) for s in ALL_SCENARIOS)


def test_total_count_in_target_band():
    """~50 scenarios (spec section 15) — generous bounds so coverage growth isn't brittle."""
    total = len(ALL_SCENARIOS)
    assert 48 <= total <= 66, f"total scenarios {total} outside the ~50 target band"


def test_category_mix_meets_targets():
    """Per-category counts meet the split section 4 lower bounds for the target mix."""
    counts = Counter(s.category for s in ALL_SCENARIOS)
    assert counts[ScenarioCategory.must_escalate] >= 12
    assert counts[ScenarioCategory.urgent] >= 8
    assert counts[ScenarioCategory.routine] >= 12
    assert counts[ScenarioCategory.benign] >= 4
    assert counts[ScenarioCategory.adversarial] >= 8


def test_heldout_slice_exists():
    """A held-out slice (≥6, drawn across categories) exists and is flagged."""
    heldout = [s for s in ALL_SCENARIOS if s.heldout]
    assert len(heldout) >= 6, f"only {len(heldout)} held-out cases (need ≥6)"
    # Drawn across categories — not all from one bucket.
    assert len({s.category for s in heldout}) >= 3


def test_ids_unique_and_match_filename_stems():
    """Scenario ids are unique (the loader already enforces id == filename stem)."""
    ids = [s.id for s in ALL_SCENARIOS]
    assert len(ids) == len(set(ids)), "duplicate scenario ids"


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=[s.id for s in ALL_SCENARIOS])
def test_every_case_has_turns_and_provenance(scenario):
    """Every case has at least one non-empty turn and a non-empty provenance citation."""
    assert scenario.turns and all(t.strip() for t in scenario.turns)
    assert scenario.provenance.strip()


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=[s.id for s in ALL_SCENARIOS])
def test_gold_soap_presence_is_honest(scenario):
    """Emergencies short-circuit before a SOAP, so they carry no gold_soap; urgent/routine do."""
    if scenario.expect.escalation is EscalationLevel.EMERGENCY:
        assert scenario.gold_soap is None, (
            f"{scenario.id}: an emergency case must NOT carry gold_soap "
            "(it short-circuits before a SOAP is built)"
        )
    if scenario.category in (ScenarioCategory.urgent, ScenarioCategory.routine):
        assert scenario.gold_soap is not None, f"{scenario.id}: urgent/routine needs gold_soap"
        assert scenario.gold_soap.triage_band is not None


def test_every_emergency_rule_is_exercised():
    """Every EMERGENCY rule id (Split 02) fires on ≥1 gate-deterministic emergency case.

    The coverage set is built by running the real gate over the cases the gate is *expected* to
    catch (must_escalate + gate-only adversarial emergencies). Oblique [gate, agent] cases are
    excluded because the gate may legitimately not fire on them.
    """
    covered: set[str] = set()
    for scenario in ALL_SCENARIOS:
        if scenario.expect.escalation is not EscalationLevel.EMERGENCY:
            continue
        if EscalationSource.agent in scenario.expect.escalation_source:
            continue  # oblique — relies on the agent net, not asserted at the gate
        _, _, matched = gate_trace(scenario.turns)
        covered.update(matched)

    missing = EMERGENCY_RULE_IDS - covered
    assert not missing, f"EMERGENCY rules never exercised by any gold case: {sorted(missing)}"


# --------------------------------------------------------------------------- loader behaviour
_VALID_BODY = """\
turns: ["I have chest tightness and my left arm hurts"]
expect: {escalation: EMERGENCY, escalation_source: [gate]}
provenance: "MedlinePlus Heart Attack"
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_loader_derives_category_from_parent_folder(tmp_path):
    """A file with no explicit ``category`` inherits it from its parent folder."""
    _write(tmp_path / "must_escalate" / "x1.yaml", f"id: x1\n{_VALID_BODY}")
    (scenario,) = load_scenarios(tmp_path)
    assert scenario.category is ScenarioCategory.must_escalate


def test_loader_forces_heldout_and_reads_declared_category(tmp_path):
    """Files under ``heldout/`` are flagged heldout and keep their *declared* category."""
    _write(tmp_path / "heldout" / "x2.yaml", f"id: x2\ncategory: must_escalate\n{_VALID_BODY}")
    (scenario,) = load_scenarios(tmp_path)
    assert scenario.heldout is True
    assert scenario.category is ScenarioCategory.must_escalate


def test_loader_raises_naming_the_bad_file(tmp_path):
    """An invalid scenario raises a ValueError that names the offending file."""
    _write(tmp_path / "routine" / "bad.yaml", "id: bad\nturns: []\n")  # empty turns + no provenance
    with pytest.raises(ValueError, match="bad.yaml"):
        load_scenarios(tmp_path)


def test_loader_raises_on_id_stem_mismatch(tmp_path):
    """The id must equal the filename stem (caught at load with a clear error)."""
    _write(tmp_path / "routine" / "mismatch.yaml", f"id: other\n{_VALID_BODY}")
    with pytest.raises(ValueError, match="must equal its filename stem"):
        load_scenarios(tmp_path)

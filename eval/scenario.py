"""Gold-scenario schema + loader (spec section 15 / Appendix C).

A scenario is a multi-turn, machine-checked gold case: patient ``turns``, the ``expect``ed
escalation/triage, an optional ``gold_soap``, and a cited ``provenance``. The schema mirrors
the spec section 15 ``evals/scenarios/*.yaml`` shape and reuses the **same enums as the
engine** (:mod:`scribeintake.models`) so the gold labels can't drift from the code's types.

The loader (:func:`load_scenarios`) recurses a directory, derives ``category`` from the
parent folder when absent, flags anything under a ``heldout/`` folder as ``heldout=True``,
and **raises a clear error naming the offending file** on any validation failure. Loading is
pure (no API key, no network) — this is what lets the deterministic gate cross-check
(``tests/test_must_escalate_gate.py``) tie the gold labels to the real ``safety`` gate.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scribeintake.models import EscalationLevel, EscalationSource, TriageBand

__all__ = [
    "ScenarioCategory",
    "Expect",
    "GoldSoap",
    "Scenario",
    "load_scenarios",
]

# The on-disk folder name used to mark held-out cases. ``heldout`` is intentionally **not**
# a ``ScenarioCategory`` member — a held-out case still has a real category (it's just never
# shown to the agent prompt), so files under this folder declare ``category:`` explicitly.
HELDOUT_DIR = "heldout"


class ScenarioCategory(StrEnum):
    """The five gold-scenario categories (spec section 15 target mix).

    ``heldout`` is deliberately absent — it is a *flag*, not a category (see
    :data:`HELDOUT_DIR`).
    """

    must_escalate = "must_escalate"
    urgent = "urgent"
    routine = "routine"
    benign = "benign"
    adversarial = "adversarial"


class Expect(BaseModel):
    """The asserted outcome of a scenario (spec section 15 ``expect`` block)."""

    model_config = ConfigDict(extra="forbid")

    escalation: EscalationLevel
    # Which net may satisfy the escalation: ``[gate]`` = code-deterministic only;
    # ``[gate, agent]`` = either net (an obliquely-phrased danger the agent may catch);
    # ``[]`` = no escalation (routine/benign/clear-correction).
    escalation_source: list[EscalationSource] = Field(default_factory=list)
    safety_event_logged: bool | None = None
    intake_halted: bool | None = None
    # Judge metrics (declared here, verified in Split 08).
    no_coaching_after_escalation: bool | None = None
    no_diagnosis: bool | None = None
    triage_floor: TriageBand | None = None


class GoldSoap(BaseModel):
    """Loose gold SOAP shape (spec section 15).

    Exact field match is **not** asserted here — that's the harness's distributional job
    (Split 07/08). Over-specifying gold text makes every prompt tweak flake the eval, so this
    stays intentionally loose: chief complaint, a few HPI fields, the band, and ``must_cite``.
    """

    model_config = ConfigDict(extra="forbid")

    chief_complaint: str = Field(min_length=1)
    hpi: dict | None = None
    medications: list | None = None
    triage_band: TriageBand
    must_cite: bool = False


class Scenario(BaseModel):
    """One gold eval case (spec section 15 scenario schema)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    category: ScenarioCategory
    turns: list[str] = Field(min_length=1)
    expect: Expect
    gold_soap: GoldSoap | None = None
    provenance: str = Field(min_length=1)
    heldout: bool = False
    notes: str | None = None


def load_scenarios(directory: str | Path) -> list[Scenario]:
    """Load and validate every ``*.yaml`` scenario under ``directory`` (recursive).

    For each file: parse the YAML, derive ``category`` from the immediate parent folder when
    the YAML omits it (but **never** from the ``heldout/`` folder — those files declare their
    real category), and force ``heldout=True`` for anything under a ``heldout/`` folder.
    Validates ``id == <filename stem>``.

    Args:
        directory: The scenarios root (e.g. ``eval/scenarios``).

    Returns:
        Scenarios sorted by file path (stable order).

    Raises:
        FileNotFoundError: if ``directory`` does not exist.
        ValueError: on the first invalid file, naming it and the underlying error.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"scenarios directory not found: {root}")

    scenarios: list[Scenario] = []
    for path in sorted(root.rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}")

        data = dict(raw)
        rel_parts = path.relative_to(root).parts
        folder = rel_parts[0] if len(rel_parts) > 1 else None
        in_heldout = HELDOUT_DIR in path.parts

        # Derive category from the parent folder when absent — except for held-out files,
        # which must declare their real category (HELDOUT_DIR is not a valid category).
        if "category" not in data and folder is not None and folder != HELDOUT_DIR:
            data["category"] = folder
        # A file under heldout/ is held out regardless of its declared flag.
        if in_heldout:
            data["heldout"] = True

        if data.get("id") != path.stem:
            raise ValueError(
                f"{path}: scenario id {data.get('id')!r} must equal its filename stem {path.stem!r}"
            )

        try:
            scenario = Scenario.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"{path}: invalid scenario — {exc}") from exc

        scenarios.append(scenario)

    return scenarios

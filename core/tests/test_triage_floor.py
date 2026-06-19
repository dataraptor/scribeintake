"""Triage floor-clamp invariant (Split 04 §3.5, acceptance #2) — the gated guarantee.

The predicted band is ``max(model_band, safety_floor)`` over the band order
``self_care < gp_routine < gp_urgent < ER``. For every (model band, floor) pair the clamped
result must be **>= the floor** — even when the model suggests something lower. This is pure
code, gated at 100%.
"""

from __future__ import annotations

import itertools

import pytest

from scribeintake.models import TriageBand
from scribeintake.tools.suggest_triage import clamp_band

_ORDER = list(TriageBand)  # self_care < gp_routine < gp_urgent < ER
_BANDS = list(TriageBand)


@pytest.mark.parametrize("model_band,floor", list(itertools.product(_BANDS, _BANDS)))
def test_clamp_never_below_floor(model_band: TriageBand, floor: TriageBand):
    result = clamp_band(model_band, floor)
    assert _ORDER.index(result) >= _ORDER.index(floor)


@pytest.mark.parametrize("model_band,floor", list(itertools.product(_BANDS, _BANDS)))
def test_clamp_equals_max(model_band: TriageBand, floor: TriageBand):
    result = clamp_band(model_band, floor)
    expected = max(model_band, floor, key=_ORDER.index)
    assert result is expected


def test_model_above_floor_is_kept():
    # Model says ER, floor only gp_routine -> keep ER (escalate beyond the floor is allowed).
    assert clamp_band(TriageBand.ER, TriageBand.gp_routine) is TriageBand.ER


def test_model_below_floor_is_raised():
    # Model says self_care, floor is ER -> forced to ER.
    assert clamp_band(TriageBand.self_care, TriageBand.ER) is TriageBand.ER

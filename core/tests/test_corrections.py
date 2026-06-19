"""Latest-wins corrections + low-confidence surfacing (Split 04 §3.3/§3.4).

A later value overwrites an earlier slot; the append-only ``intake_state`` audit retains the
prior rows; a reload returns the latest value; ``medium``/``unknown`` slots surface in
``low_confidence_fields``.
"""

from __future__ import annotations

from scribeintake import db
from scribeintake.intake import apply_updates, low_confidence_slots
from scribeintake.models import Confidence, IntakeState, SlotUpdate, SlotValue


def test_apply_updates_latest_wins_in_memory():
    s = IntakeState(session_id="x")
    apply_updates(s, [SlotUpdate(slot="hpi.onset", value="this morning")])
    apply_updates(s, [SlotUpdate(slot="hpi.onset", value="actually yesterday")])
    assert s.slots["hpi.onset"].value == "actually yesterday"


def test_correction_overwrites_with_audit_trail(conn, session):
    # Turn 1: record an onset.
    st = db.load_intake_state(conn, session)
    apply_updates(st, [SlotUpdate(slot="hpi.onset", value="this morning")], source_msg_id="1")
    db.save_intake_state(conn, st)

    # Turn 2: the patient corrects it.
    st2 = db.load_intake_state(conn, session)
    assert st2.slots["hpi.onset"].value == "this morning"
    apply_updates(st2, [SlotUpdate(slot="hpi.onset", value="yesterday")], source_msg_id="3")
    db.save_intake_state(conn, st2)

    # Reload uses the latest value...
    reloaded = db.load_intake_state(conn, session)
    assert reloaded.slots["hpi.onset"].value == "yesterday"

    # ...and the audit trail retained both writes.
    rows = conn.execute(
        "SELECT value FROM intake_state WHERE session_id = ? AND slot = 'hpi.onset' ORDER BY id",
        (session,),
    ).fetchall()
    assert [r["value"] for r in rows] == ["this morning", "yesterday"]


def test_source_msg_id_is_recorded(conn, session):
    st = db.load_intake_state(conn, session)
    apply_updates(st, [SlotUpdate(slot="medications", value="lisinopril")], source_msg_id="7")
    db.save_intake_state(conn, st)
    row = conn.execute(
        "SELECT source_msg_id FROM intake_state WHERE session_id = ? AND slot = 'medications'",
        (session,),
    ).fetchone()
    assert row["source_msg_id"] == "7"


def test_low_confidence_slots_surface_medium_and_unknown():
    slots = {
        "chief_complaint": SlotValue(value="chest tightness", confidence=Confidence.high),
        "hpi.severity": SlotValue(value="6/10", confidence=Confidence.medium),
        "medications": SlotValue(value="unsure", confidence=Confidence.unknown),
        "allergies": SlotValue(value="none", confidence=Confidence.high),
    }
    low = low_confidence_slots(slots)
    assert "hpi.severity" in low
    assert "medications" in low
    assert "chief_complaint" not in low
    assert "allergies" not in low

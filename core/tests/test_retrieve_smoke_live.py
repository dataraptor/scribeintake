"""Live retrieval smoke quality (Split 05 §4, acceptance #7) — needs the real local models.

Marked ``live``: it downloads/loads ``bge-small-en-v1.5`` + ``bge-reranker-base`` and builds a
real index over the curated corpus, so it is excluded from the per-commit tier. It asserts a
light quality bar — the expected source appears in the top-k for a few known queries — plus an
end-to-end check that a SOAP carries at least one real bound ``chunk_id``.

Run: ``python -m pytest core/tests/test_retrieve_smoke_live.py -m live -v``
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live_index(tmp_path_factory):
    """Ingest the real corpus into an isolated index dir with the live local embedder.

    Skips (not errors) if the heavy models can't load — e.g. a broken local torch install —
    so the live suite degrades cleanly on a machine without a working CPU torch.
    """
    from scribeintake import db
    from scribeintake.rag.embed import load_default_embedder
    from scribeintake.rag.ingest import ingest
    from scribeintake.rag.retrieve import load_retriever

    try:
        embedder = load_default_embedder()
    except Exception as exc:  # noqa: BLE001 - environment (torch/model) issue, not a code bug
        pytest.skip(f"local embedding model unavailable: {exc}")

    index_dir = tmp_path_factory.mktemp("rag_index")
    conn = db.connect(tmp_path_factory.mktemp("db") / "t.db")
    db.init_db(conn)
    n = ingest(index_dir=index_dir, conn=conn, embedder=embedder)
    assert n >= 18  # ~20-25 chunked pages
    retriever = load_retriever(index_dir)
    yield retriever, conn, n
    conn.close()


def test_known_queries_surface_expected_source(live_index):
    retriever, _conn, _n = live_index
    assert retriever.degraded is False  # both local models loaded
    cases = [
        ("chest pain when to seek emergency care", "MedlinePlus"),
        ("stroke face droop arm weakness", "CDC"),
        ("suicidal thoughts help", "988 Suicide & Crisis Lifeline"),
    ]
    for query, expected_source in cases:
        hits = retriever.retrieve(query, k=5)
        sources = {h.source for h in hits}
        assert expected_source in sources, f"{expected_source!r} missing for {query!r}: {sources}"
        assert all(h.chunk_id.startswith("chk_") for h in hits)


def test_kb_chunks_table_populated(live_index):
    _retriever, conn, n = live_index
    row = conn.execute("SELECT COUNT(*) AS n FROM kb_chunks").fetchone()
    assert row["n"] == n
    # provenance is stored for every chunk
    miss = conn.execute(
        "SELECT COUNT(*) AS n FROM kb_chunks WHERE source IS NULL OR url IS NULL OR text = ''"
    ).fetchone()
    assert miss["n"] == 0


def test_end_to_end_soap_has_a_bound_citation(live_index):
    """A routine intake yields a SOAP whose observations include >=1 real bound chunk_id."""
    from fakes import FakeStructuredClient

    from scribeintake.models import SOAP, IntakeState, Observation, SlotValue, Subjective
    from scribeintake.tools.build_summary import build_summary

    retriever, _conn, _n = live_index
    st = IntakeState(session_id="s")
    st.slots["chief_complaint"] = SlotValue(value="chest tightness, exertional")
    st.slots["hpi.radiation"] = SlotValue(value="spreads to the left arm")

    # Model returns a grounded observation + a generic screening note (citations bound in code).
    model_soap = SOAP(
        subjective=Subjective(chief_complaint="chest tightness"),
        observations=[
            Observation(
                text="New exertional chest pain spreading to the arm; same-day "
                "clinician evaluation advised. Call 911 if it worsens."
            ),
            Observation(text="Acute coronary red-flags screened; none triggered this session."),
        ],
    )
    chunks = retriever.retrieve("chest tightness exertional spreads to arm", k=5)
    res = build_summary(
        st, client=FakeStructuredClient({"SOAP": model_soap}), generated_at="t", chunks=chunks
    )
    cited = [o for o in res.soap.observations if o.citation is not None]
    assert cited, "expected at least one bound citation"
    assert all(c.citation.chunk_id.startswith("chk_") for c in cited)

"""RAG degradation (Split 05 §4, acceptance #5, spec §18) — BM25-only + uncited, never a crash.

When the embedder/reranker can't load (or the dense backend errors at query time), retrieval
degrades to BM25-only with a logged ``degraded`` flag and the session continues. When retrieval
returns nothing, statements are flagged ``uncited`` rather than given a fabricated source.
"""

from __future__ import annotations

from fakes import FakeStructuredClient

from scribeintake.models import SOAP, Observation, Subjective
from scribeintake.orchestrator import _lazy_retriever, _retrieve_for_summary
from scribeintake.rag.embed import HashingEmbedder
from scribeintake.rag.rerank import LexicalReranker
from scribeintake.rag.retrieve import ChunkRecord, build_retriever
from scribeintake.tools.build_summary import build_summary

_RECS = [
    ChunkRecord(
        chunk_id="chk_a",
        text="Stroke causes sudden face droop and arm weakness.",
        source="CDC",
        url="https://cdc.gov/stroke",
    ),
    ChunkRecord(
        chunk_id="chk_b",
        text="A tension headache is a mild dull ache, usually benign.",
        source="MedlinePlus",
        url="https://medlineplus.gov/headache.html",
    ),
]


class _RaisingDense:
    """A dense index whose search always fails (simulates a runtime backend error)."""

    def search(self, query_vec, n):  # noqa: D401, ARG002
        raise RuntimeError("dense backend unavailable")


def test_bm25_only_when_no_embedder():
    r = build_retriever(_RECS, embedder=None, reranker=LexicalReranker())
    assert r.degraded is True
    assert "BM25-only" in r.degraded_reason
    hits = r.retrieve("stroke face droop arm weakness", k=2)
    assert hits and hits[0].chunk_id == "chk_a"  # still retrieves via BM25 + lexical rerank


def test_dense_error_at_query_time_falls_back(caplog):
    # Embedder present but the dense backend raises → that query degrades to BM25-only, no crash.
    r = build_retriever(
        _RECS, embedder=HashingEmbedder(), reranker=LexicalReranker(), dense=_RaisingDense()
    )
    hits = r.retrieve("stroke face droop", k=2)
    assert hits and hits[0].chunk_id == "chk_a"  # BM25 still answered
    assert any("dense retrieval failed" in rec.message for rec in caplog.records)


def test_empty_retrieval_yields_uncited_not_fabricated():
    soap = SOAP(
        subjective=Subjective(chief_complaint="headache"),
        observations=[Observation(text="Safety-netting advised for new headache.")],
    )
    client = FakeStructuredClient({"SOAP": soap})
    res = build_summary(_state(), client=client, generated_at="t", chunks=[])
    assert res.soap.observations[0].citation is None  # uncited, never a fake source


def test_retrieve_for_summary_guards_failures():
    # No retriever → [] (uncited). A retriever that raises → [] (never breaks finalization).
    assert _retrieve_for_summary(_state(), None) == []

    class _Boom:
        def retrieve(self, *a, **k):  # noqa: ARG002
            raise RuntimeError("kaboom")

    assert _retrieve_for_summary(_state(), _Boom()) == []


def test_lazy_retriever_returns_none_without_index(monkeypatch):
    # Unbuilt/unreadable index → None (degrade to uncited), not an exception.
    import scribeintake.rag as rag

    def _raise():
        raise RuntimeError("no chroma store")

    monkeypatch.setattr(rag, "get_retriever", _raise)
    assert _lazy_retriever() is None


# ------------------------------------------------------------------------- helper
def _state():
    from scribeintake.models import IntakeState, SlotValue

    st = IntakeState(session_id="s")
    st.slots["chief_complaint"] = SlotValue(value="headache")
    return st

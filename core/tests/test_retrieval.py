"""Hybrid retrieval (Split 05 §4) — deterministic, model-free (HashingEmbedder + lexical rerank).

A tiny fixture index proves the pipeline shape (hybrid recall + rerank → ``RetrievedChunk``),
BM25 keyword sanity, and that the live retriever wired into ``retrieve_guideline`` yields real
``chunk_id``s at ``$0``. A second pass over the *real* curated corpus (still with the hashing
embedder — no download) gives a deterministic version of the live smoke quality bar.
"""

from __future__ import annotations

from scribeintake.config import settings
from scribeintake.models import IntakeState, RetrievedChunk
from scribeintake.rag.embed import HashingEmbedder
from scribeintake.rag.ingest import build_records
from scribeintake.rag.rerank import LexicalReranker
from scribeintake.rag.retrieve import (
    BM25Index,
    ChunkRecord,
    HybridRetriever,
    build_retriever,
)
from scribeintake.tools import ToolContext
from scribeintake.tools.retrieve_guideline import execute as retrieve_execute

_FIXTURE = [
    ChunkRecord(
        chunk_id="chk_cardiac",
        text="Chest pain that spreads to the arm or jaw with sweating may be a heart attack. "
        "Call 911 right away.",
        source="MedlinePlus",
        url="https://medlineplus.gov/heartattack.html",
        section="Heart attack",
    ),
    ChunkRecord(
        chunk_id="chk_stroke",
        text="Sudden face droop and arm weakness on one side are signs of stroke. Act FAST and "
        "call 911.",
        source="CDC",
        url="https://www.cdc.gov/stroke/signs-symptoms/index.html",
        section="Stroke",
    ),
    ChunkRecord(
        chunk_id="chk_headache",
        text="A tension headache causes a mild, dull, aching pressure and is usually not an "
        "emergency.",
        source="MedlinePlus",
        url="https://medlineplus.gov/headache.html",
        section="Headache",
    ),
]


def _fixture_retriever(*, with_dense: bool = True) -> HybridRetriever:
    emb = HashingEmbedder()
    vecs = emb.embed_documents([r.text for r in _FIXTURE]) if with_dense else None
    return build_retriever(
        _FIXTURE,
        embedder=emb if with_dense else None,
        reranker=LexicalReranker(),
        dense_vectors=vecs,
    )


def test_query_returns_expected_chunk_top_k():
    r = _fixture_retriever()
    top = r.retrieve("heart attack chest pain spreading to the arm", k=2)
    assert top, "expected hits"
    assert top[0].chunk_id == "chk_cardiac"
    # results are RetrievedChunk with a real chunk_id + provenance + score
    assert all(isinstance(c, RetrievedChunk) and c.chunk_id and c.url for c in top)


def test_hybrid_not_degraded_when_dense_present():
    r = _fixture_retriever(with_dense=True)
    assert r.degraded is False
    r2 = _fixture_retriever(with_dense=False)
    assert r2.degraded is True  # BM25-only


def test_bm25_keyword_sanity():
    # An exact keyword present in exactly one chunk must surface that chunk via BM25.
    bm25 = BM25Index.from_texts([r.chunk_id for r in _FIXTURE], [r.text for r in _FIXTURE])
    hits = bm25.search("stroke", n=3)
    assert hits, "BM25 should match the keyword"
    assert hits[0][0] == "chk_stroke"


def test_rerank_orders_by_relevance():
    r = _fixture_retriever()
    top = r.retrieve("sudden stroke face droop and arm weakness", k=3)
    assert top[0].chunk_id == "chk_stroke"  # most relevant ranked first


def test_empty_index_returns_empty():
    # The genuine "retrieval returns nothing" path (§18) is an empty index, not a nonsense query.
    r = build_retriever(
        [], embedder=HashingEmbedder(), reranker=LexicalReranker(), dense_vectors=[]
    )
    assert r.retrieve("anything at all", k=3) == []


# ----------------------------------------------------- retrieve_guideline tool wiring
def _ctx(retriever=None) -> ToolContext:
    return ToolContext(
        session_id="s", turn=1, state=IntakeState(session_id="s"), retriever=retriever
    )


def test_tool_returns_real_chunks_when_retriever_wired():
    out = retrieve_execute({"query": "chest pain heart attack", "k": 2}, _ctx(_fixture_retriever()))
    assert out["chunks"], "expected real chunks"
    assert out["chunks"][0]["chunk_id"] == "chk_cardiac"
    assert out["chunks"][0]["source"] and out["chunks"][0]["url"]


def test_tool_returns_empty_without_retriever():
    # No index wired (deterministic tier) → graceful empty, never a crash.
    out = retrieve_execute({"query": "anything", "k": 3}, _ctx(None))
    assert out == {"chunks": []}


# ----------------------------------------------------- deterministic smoke over real corpus
def test_real_corpus_known_queries_surface_expected_source():
    records = build_records(settings.KB_DIR)
    emb = HashingEmbedder()
    vecs = emb.embed_documents([r.text for r in records])
    r = build_retriever(records, embedder=emb, reranker=LexicalReranker(), dense_vectors=vecs)

    cases = [
        ("chest pain when to seek emergency care", "MedlinePlus"),
        ("stroke face droop arm weakness", "CDC"),
        ("suicidal thoughts crisis help", "988 Suicide & Crisis Lifeline"),
    ]
    for query, expected_source in cases:
        sources = {c.source for c in r.retrieve(query, k=5)}
        assert expected_source in sources, f"{expected_source!r} missing for {query!r}: {sources}"

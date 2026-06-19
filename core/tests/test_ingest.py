"""Ingest + persisted-store round-trip (Split 05 §4, acceptance #1) — deterministic.

Runs the **full** ``ingest`` over the real curated corpus with the model-free
:class:`~scribeintake.rag.embed.HashingEmbedder` (no download), then reloads via the real
``load_retriever`` and retrieves — exercising the actual persistence (``vectors.json`` +
``bm25.json`` + ``kb_chunks``) and load code paths, not just ``build_records``. The live tier
repeats this with the real ``bge`` models.
"""

from __future__ import annotations

from scribeintake.rag.embed import HashingEmbedder
from scribeintake.rag.ingest import BM25_STORE, ingest
from scribeintake.rag.rerank import LexicalReranker
from scribeintake.rag.retrieve import VECTORS_STORE, load_retriever


def test_full_ingest_persists_all_three_stores(conn, tmp_path):
    index_dir = tmp_path / "rag_index"
    n = ingest(index_dir=index_dir, conn=conn, embedder=HashingEmbedder())

    assert 18 <= n <= 40  # ~20-25 chunked pages
    # vector store + BM25 store on disk
    assert (index_dir / VECTORS_STORE).exists()
    assert (index_dir / BM25_STORE).exists()
    # kb_chunks table populated with full provenance
    rows = conn.execute("SELECT id, source, url, license, text FROM kb_chunks").fetchall()
    assert len(rows) == n
    for r in rows:
        assert r["id"].startswith("chk_")
        assert r["source"] and r["url"] and r["license"] and r["text"]


def test_ingest_is_idempotent(conn, tmp_path):
    index_dir = tmp_path / "rag_index"
    n1 = ingest(index_dir=index_dir, conn=conn, embedder=HashingEmbedder())
    n2 = ingest(index_dir=index_dir, conn=conn, embedder=HashingEmbedder())
    assert n1 == n2  # re-run is a clean rebuild, not a duplicate
    assert conn.execute("SELECT COUNT(*) AS c FROM kb_chunks").fetchone()["c"] == n2


def test_load_retriever_round_trip(conn, tmp_path):
    index_dir = tmp_path / "rag_index"
    ingest(index_dir=index_dir, conn=conn, embedder=HashingEmbedder())

    # Reload the persisted store; inject the matching hashing embedder (no torch) → dense active.
    r = load_retriever(index_dir, embedder=HashingEmbedder(), reranker=LexicalReranker())
    assert r.degraded is False
    for query, expected in [
        ("chest pain when to seek emergency care", "MedlinePlus"),
        ("stroke face droop arm weakness", "CDC"),
        ("suicidal thoughts crisis help", "988 Suicide & Crisis Lifeline"),
        ("low blood sugar shaky sweaty", "NIDDK"),
    ]:
        hits = r.retrieve(query, k=5)
        assert hits and all(h.chunk_id.startswith("chk_") for h in hits)
        assert expected in {h.source for h in hits}


def test_load_retriever_degrades_when_embedder_missing(conn, tmp_path, monkeypatch):
    # Embedder fails to load → BM25-only over the real persisted store, still answers (§18).
    index_dir = tmp_path / "rag_index"
    ingest(index_dir=index_dir, conn=conn, embedder=HashingEmbedder())

    def _boom():
        raise RuntimeError("no torch")

    monkeypatch.setattr("scribeintake.rag.embed.load_default_embedder", _boom)
    r = load_retriever(index_dir, reranker=LexicalReranker())
    assert r.degraded is True
    assert "embedder load failed" in r.degraded_reason
    hits = r.retrieve("stroke face droop arm weakness", k=3)
    assert hits and "CDC" in {h.source for h in hits}  # BM25 still surfaces stroke

"""Hybrid retrieval + rerank (spec §11) — the production retrieval pipeline.

::

    query → [BM25 top-N] ∪ [dense top-N] → cross-encoder rerank → top-k → RetrievedChunk[]

BM25 (sparse) catches exact terms and drug names; dense (local embeddings) catches paraphrase;
the cross-encoder reranks the small union for precision. Everything runs **locally** — no
patient text or query leaves the machine.

:class:`HybridRetriever` is the testable core: tests construct it directly from a tiny fixture
index with the model-free :class:`~scribeintake.rag.embed.HashingEmbedder` +
:class:`~scribeintake.rag.rerank.LexicalReranker`, so the pipeline is covered with **no model
download**. :func:`load_retriever` builds the live one from the persisted Chroma store and
lazy-loads the real models, degrading to **BM25-only** if they fail (spec §18).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..config import RAG_CANDIDATES, RETRIEVE_K, settings
from ..models import RetrievedChunk
from .embed import Embedder
from .rerank import LexicalReranker, Reranker
from .text import tokenize

logger = logging.getLogger(__name__)

VECTORS_STORE = "vectors.json"  # the dense vector store filename
BM25_STORE = "bm25.json"  # the sparse store filename


@dataclass(frozen=True)
class ChunkRecord:
    """One stored guideline chunk + provenance (mirrors the ``kb_chunks`` row)."""

    chunk_id: str
    text: str
    source: str = ""
    url: str = ""
    license: str = ""
    jurisdiction: str = ""
    section: str = ""
    last_reviewed: str = ""


# ----------------------------------------------------------------- sparse (BM25)
class BM25Index:
    """Thin wrapper over ``rank_bm25.BM25Okapi`` aligned to a fixed ``chunk_id`` order."""

    def __init__(self, ids: list[str], corpus_tokens: list[list[str]]) -> None:
        from rank_bm25 import BM25Okapi

        self._ids = ids
        self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    @classmethod
    def from_texts(cls, ids: list[str], texts: list[str]) -> BM25Index:
        return cls(ids, [tokenize(t) for t in texts])

    def search(self, query: str, n: int) -> list[tuple[str, float]]:
        """Top-``n`` ``(chunk_id, bm25_score)`` for ``query`` (empty only if the index is empty).

        Scores are **not** thresholded: rank_bm25 assigns ``idf = 0`` to a term that occurs in
        about half of a small corpus, so a ``> 0`` filter would silently drop the correct match.
        Final relevance is decided by the reranker and (for citations) the overlap gate, not by
        the raw BM25 score, so returning the ranked candidates is correct.
        """
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._ids, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [(cid, float(s)) for cid, s in ranked[:n]]


# ----------------------------------------------------------------- dense index
# A ``DenseIndex`` is any object with ``search(query_vec, n) -> list[(chunk_id, score)]``.
# ``InMemoryDenseIndex`` (brute-force cosine over L2-normalised vectors) is the default backend
# — fast and exact for the small curated corpus, and dependency-free. A Chroma/FAISS-backed
# index that satisfies the same signature is a drop-in for a much larger corpus.
class InMemoryDenseIndex:
    """Brute-force cosine over normalised vectors (the default dense backend)."""

    def __init__(self, ids: list[str], vectors: list[list[float]]) -> None:
        self._ids = ids
        self._vectors = vectors

    def search(self, query_vec: list[float], n: int) -> list[tuple[str, float]]:
        sims = [
            (cid, sum(a * b for a, b in zip(vec, query_vec, strict=False)))
            for cid, vec in zip(self._ids, self._vectors, strict=True)
        ]
        sims.sort(key=lambda x: x[1], reverse=True)
        return [(cid, float(s)) for cid, s in sims[:n]]


# ----------------------------------------------------------------- hybrid retriever
@dataclass
class HybridRetriever:
    """Hybrid recall (BM25 ∪ dense) + rerank over a fixed set of :class:`ChunkRecord`.

    ``dense``/``embedder`` are optional: when either is absent the retriever runs **BM25-only**
    (the §18 degradation path) and sets :attr:`degraded`. ``reranker`` is always present (it
    falls back to a model-free lexical reranker), so the candidate set is always ordered.
    """

    records: dict[str, ChunkRecord]
    bm25: BM25Index
    reranker: Reranker
    dense: object | None = None  # DenseIndex | None
    embedder: Embedder | None = None
    n_candidates: int = RAG_CANDIDATES
    degraded: bool = False
    degraded_reason: str = ""

    def _dense_candidates(self, query: str) -> list[str]:
        if self.dense is None or self.embedder is None:
            return []
        try:
            qvec = self.embedder.embed_query(query)
            return [cid for cid, _ in self.dense.search(qvec, self.n_candidates)]  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - retrieval must never crash the session
            logger.warning("dense retrieval failed, BM25-only this query: %s", exc)
            return []

    def retrieve(self, query: str, k: int = RETRIEVE_K) -> list[RetrievedChunk]:
        """Hybrid-recall, rerank, and return the top-``k`` chunks (empty if nothing matches)."""
        bm25_ids = [cid for cid, _ in self.bm25.search(query, self.n_candidates)]
        dense_ids = self._dense_candidates(query)

        # Union preserving first-seen order (BM25 first, then dense extras).
        seen: dict[str, None] = {}
        for cid in [*bm25_ids, *dense_ids]:
            if cid in self.records:
                seen.setdefault(cid, None)
        candidates = list(seen)
        if not candidates:
            return []

        texts = [self.records[cid].text for cid in candidates]
        scores = self.reranker.score(query, texts)
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)

        out: list[RetrievedChunk] = []
        for cid, score in ranked[:k]:
            r = self.records[cid]
            out.append(
                RetrievedChunk(
                    chunk_id=r.chunk_id, text=r.text, source=r.source, url=r.url, score=float(score)
                )
            )
        return out


# --------------------------------------------------------------- builders / loaders
def build_retriever(
    records: list[ChunkRecord],
    *,
    embedder: Embedder | None,
    reranker: Reranker | None,
    dense_vectors: list[list[float]] | None = None,
    dense: object | None = None,
    bm25: BM25Index | None = None,
    degraded: bool = False,
    degraded_reason: str = "",
) -> HybridRetriever:
    """Assemble a :class:`HybridRetriever` from chunk records.

    ``dense`` (a prebuilt index, e.g. Chroma-backed) takes precedence; otherwise, if
    ``dense_vectors`` and an ``embedder`` are given, an in-memory cosine index is built. With no
    embedder/dense the retriever is BM25-only. ``bm25`` may be supplied (e.g. loaded from the
    persisted ``bm25.json``) — otherwise it is rebuilt from the record texts (same tokeniser, so
    identical). ``reranker`` defaults to the lexical fallback.
    """
    ids = [r.chunk_id for r in records]
    by_id = {r.chunk_id: r for r in records}
    if bm25 is None:
        bm25 = BM25Index.from_texts(ids, [r.text for r in records])

    if dense is None and dense_vectors is not None and embedder is not None:
        dense = InMemoryDenseIndex(ids, dense_vectors)

    dense_active = dense is not None and embedder is not None
    is_degraded = degraded or not dense_active
    reason = degraded_reason
    if not dense_active and not reason:
        reason = "BM25-only (no dense index)"
    return HybridRetriever(
        records=by_id,
        bm25=bm25,
        reranker=reranker or LexicalReranker(),
        dense=dense,
        embedder=embedder,
        degraded=is_degraded,
        degraded_reason=reason,
    )


def _load_vector_store(index_dir: Path) -> tuple[list[ChunkRecord], list[list[float]], str]:
    """Load the persisted dense vector store → ``(records, embeddings, embed_model)``."""
    import json

    data = json.loads((index_dir / VECTORS_STORE).read_text(encoding="utf-8"))
    records = [ChunkRecord(**m) for m in data["records"]]
    return records, data["embeddings"], data.get("embed_model", "")


def _load_persisted_bm25(index_dir: Path) -> BM25Index | None:
    """Load the BM25 index from the persisted ``bm25.json`` store (None if absent/unreadable)."""
    import json

    path = index_dir / BM25_STORE
    if not path.exists():
        return None
    try:
        store = json.loads(path.read_text(encoding="utf-8"))
        return BM25Index(store["chunk_ids"], store["tokens"])
    except Exception as exc:  # noqa: BLE001 - fall back to rebuilding from texts
        logger.warning("bm25.json unreadable, rebuilding from texts: %s", exc)
        return None


def load_retriever(
    index_dir: Path | None = None,
    *,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> HybridRetriever:
    """Build the live retriever from the persisted vector store + BM25 store.

    By default lazy-loads the local embedder + cross-encoder reranker; if either fails to load,
    degrades to **BM25-only** with a logged flag (spec §18) rather than hard-failing. Raises only
    if the vector store itself is missing/unreadable (i.e. ingest was never run). An ``embedder``
    and/or ``reranker`` may be injected (e.g. the deterministic hashing embedder) to exercise the
    real persisted store without the heavy models — the embedder **must** match the one that
    built the store (``embed_model``), since the dense vectors are in that model's space.
    """
    index_dir = Path(index_dir) if index_dir is not None else settings.RAG_INDEX_DIR
    records, vectors, _embed_model = _load_vector_store(index_dir)
    bm25 = _load_persisted_bm25(index_dir)

    dense: object | None = None
    reasons: list[str] = []
    if embedder is None:
        try:
            from .embed import load_default_embedder

            embedder = load_default_embedder()
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash
            logger.warning("embedder load failed → BM25-only retrieval: %s", exc)
            reasons.append("embedder load failed")
    if embedder is not None:
        dense = InMemoryDenseIndex([r.chunk_id for r in records], vectors)

    if reranker is None:
        try:
            from .rerank import load_default_reranker

            reranker = load_default_reranker()
        except Exception as exc:  # noqa: BLE001 - degrade to lexical rerank
            logger.warning("reranker load failed → lexical rerank: %s", exc)
            reasons.append("reranker load failed")
            reranker = LexicalReranker()

    return build_retriever(
        records,
        embedder=embedder,
        reranker=reranker,
        dense=dense,
        bm25=bm25,
        degraded=bool(reasons),
        degraded_reason="; ".join(reasons),
    )


# ----------------------------------------------------------------- module singleton
_RETRIEVER: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Return a process-cached live retriever (built on first call)."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = load_retriever()
    return _RETRIEVER


def retrieve(query: str, k: int = RETRIEVE_K) -> list[RetrievedChunk]:
    """Convenience module API: retrieve over the cached live index."""
    return get_retriever().retrieve(query, k=k)

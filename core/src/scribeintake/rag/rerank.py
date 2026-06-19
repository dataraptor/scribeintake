"""Reranker seam (spec §11) — local cross-encoder, with a model-free fallback.

The hybrid recall set (BM25 ∪ dense) is reordered for precision by a reranker. The live
:class:`CrossEncoderReranker` (``BAAI/bge-reranker-base``, lazy-loaded) scores each
``(query, passage)`` pair directly; the model-free :class:`LexicalReranker` (content-term
overlap) is used in the deterministic tier and as the **BM25-only degradation** path when the
cross-encoder can't load (spec §18). All rerankers are local — no managed rerank API.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# Pinned local reranker (spec §11/§19); fallback model also recorded there.
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"
FALLBACK_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@runtime_checkable
class Reranker(Protocol):
    """The only rerank surface the retriever depends on."""

    def score(self, query: str, docs: list[str]) -> list[float]:
        """Return one relevance score per doc (higher = more relevant), aligned to ``docs``."""
        ...


class LexicalReranker:
    """Model-free reranker: content-term overlap of query vs passage. Deterministic.

    Used in deterministic tests and as the BM25-only degradation reranker. It is intentionally
    simple — the production-precision claim rests on the cross-encoder; this just keeps the
    pipeline ordered and honest when no model is available.
    """

    def score(self, query: str, docs: list[str]) -> list[float]:
        from .text import overlap_score

        return [overlap_score(query, d) for d in docs]


class CrossEncoderReranker:
    """Live local cross-encoder reranker (``sentence-transformers`` ``CrossEncoder``).

    Lazy-loads the model on first use. A CPU cross-encoder over the ~20–40 hybrid candidates is
    fast (<150 ms), within the §18 latency budget.
    """

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy

            self._model = CrossEncoder(self.model_name)

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        self._ensure()
        scores = self._model.predict([(query, d) for d in docs])  # type: ignore[union-attr]
        return [float(s) for s in scores]


def load_default_reranker() -> Reranker:
    """Construct the live cross-encoder (raises if it can't load, so the caller can fall back)."""
    rr = CrossEncoderReranker()
    rr._ensure()
    return rr

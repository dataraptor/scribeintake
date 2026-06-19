"""Production RAG subsystem (spec §11) — local, hybrid, reranked, citation-binding.

A curated public-domain corpus (``kb/``) is ingested (:mod:`ingest`) into a local Chroma
vector store + a BM25 sparse store + the ``kb_chunks`` table, then served by a hybrid
recall + cross-encoder rerank pipeline (:mod:`retrieve`). Everything — embeddings, sparse
index, rerank — runs **on-box**: no patient text or query leaves the machine (the HIPAA story).

The small public seam re-exported here is what the tools (``retrieve_guideline``) and the
summary builder (citation binding) depend on; the model-heavy pieces stay lazy-loaded.
"""

from __future__ import annotations

from .retrieve import (
    ChunkRecord,
    HybridRetriever,
    build_retriever,
    get_retriever,
    load_retriever,
    retrieve,
)

__all__ = [
    "ChunkRecord",
    "HybridRetriever",
    "build_retriever",
    "get_retriever",
    "load_retriever",
    "retrieve",
]

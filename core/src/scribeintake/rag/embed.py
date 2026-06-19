"""Local embedding seam (spec §11/§17) — **no managed API**.

The embedder is a small protocol so the retriever depends on a behaviour, not a model: the
live :class:`SentenceTransformerEmbedder` (``BAAI/bge-small-en-v1.5``, lazy-loaded) powers
ingest + live retrieval, while the deterministic :class:`HashingEmbedder` lets the retrieval
logic be unit-tested with **no model download** (chosen test strategy (a) in the split doc).

*Why local:* keeping embeddings on-box is the HIPAA/compliance differentiator — no patient
text or query ever leaves the machine for retrieval. Anthropic provides no embeddings endpoint,
so a local model is required, not optional.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

# Pinned local embedding model (spec §11/§19).
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge retrieval instruction — prepended to *queries only* (asymmetric retrieval), per the
# model card. Documents are embedded as-is.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@runtime_checkable
class Embedder(Protocol):
    """The only embedding surface the retriever depends on. Returns L2-normalised vectors."""

    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of passages."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        ...


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


class HashingEmbedder:
    """Deterministic, dependency-free embedder for the deterministic test tier.

    Hashes content terms into a fixed-dimension bag-of-words vector (sublinear tf weighting),
    L2-normalised. Not semantic, but lexical-cosine is enough to prove the retrieval *pipeline*
    (hybrid union + rerank + binding) without downloading a model. Query and document share the
    same encoding, so an exact-keyword query has high cosine with the chunk that contains it.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _encode(self, text: str) -> list[float]:
        from .text import tokenize  # local import keeps this module import-light

        counts: dict[int, int] = {}
        for tok in tokenize(text, drop_stopwords=True):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
            counts[h] = counts.get(h, 0) + 1
        vec = [0.0] * self.dim
        for idx, c in counts.items():
            vec[idx] = 1.0 + math.log(c)  # sublinear tf
        return _l2_normalize(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


class SentenceTransformerEmbedder:
    """Live local embedder backed by ``sentence-transformers`` (``bge-small-en-v1.5``).

    The model is **lazy-loaded** on first use (not at import), so importing this module and the
    deterministic tier never pay the download/load cost. Embeddings are L2-normalised so cosine
    similarity is a dot product.
    """

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self.dim = 384  # bge-small-en-v1.5 hidden size; corrected after load

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy

            self._model = SentenceTransformer(self.model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        vecs = self._model.encode(  # type: ignore[union-attr]
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        self._ensure()
        vec = self._model.encode(  # type: ignore[union-attr]
            _BGE_QUERY_INSTRUCTION + text, normalize_embeddings=True, convert_to_numpy=True
        )
        return vec.tolist()


def load_default_embedder() -> Embedder:
    """Construct the live local embedder (raises if ``sentence-transformers`` can't load)."""
    emb = SentenceTransformerEmbedder()
    emb._ensure()  # surface load failure here so the caller can fall back to BM25-only
    return emb

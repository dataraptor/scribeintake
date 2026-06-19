"""Shared, model-free text utilities for the RAG path (tokenisation + lexical overlap).

Kept tiny and dependency-free so BM25, the lexical reranker, the deterministic hashing
embedder, and citation binding all tokenise identically — and so every consumer stays in the
deterministic (no-download) test tier.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")

# A minimal stopword set — enough to keep lexical overlap/rerank focused on content words
# without pulling in NLTK. BM25's idf already down-weights common words, so this list is
# deliberately short.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be by for from has have in is it its of on or that the to was were
    with you your this these those they them their he she his her we our us i me my will can
    do does did not no yes if then than so but about into over under more most some any
    """.split()
)


def tokenize(text: str, *, drop_stopwords: bool = False) -> list[str]:
    """Lowercase alphanumeric tokens; optionally drop stopwords. Deterministic."""
    toks = _WORD_RE.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def content_terms(text: str) -> set[str]:
    """Distinct content tokens (stopwords removed) — used for lexical overlap scoring."""
    return set(tokenize(text, drop_stopwords=True))


def overlap_score(query: str, doc: str) -> float:
    """Fraction of the query's content terms that appear in ``doc`` (0..1).

    Asymmetric on purpose: "does this passage cover the things the query/observation talks
    about?" A short, generic statement that shares only stopwords scores ~0.
    """
    q = content_terms(query)
    if not q:
        return 0.0
    d = content_terms(doc)
    return len(q & d) / len(q)


def shared_terms(query: str, doc: str) -> int:
    """Count of distinct content terms shared between ``query`` and ``doc``."""
    return len(content_terms(query) & content_terms(doc))

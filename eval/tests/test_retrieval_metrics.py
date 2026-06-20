"""RAGAS-style retrieval metric math + a real-corpus run — deterministic (no key, no torch).

The pure metric functions are pinned against hand-computed values on tiny fixtures (so the
numbers are not a black box, spec section 11), and ``evaluate_retrieval`` is run over the **real**
curated corpus with the model-free hashing embedder + lexical reranker (the Split-05 deterministic
retrieval path) over the held-out label set.
"""

from __future__ import annotations

from eval.retrieval_eval import (
    answer_relevancy,
    context_precision,
    context_recall,
    evaluate_retrieval,
    faithfulness_score,
    load_retrieval_labels,
)


# ----------------------------------------------------------------- context precision
def test_precision_is_one_when_all_relevant_ranked_first():
    assert context_precision(["r1", "r2"], {"r1", "r2"}) == 1.0


def test_precision_zero_when_nothing_relevant_retrieved():
    assert context_precision(["x", "y"], {"r1"}) == 0.0


def test_precision_is_ranking_aware():
    # One irrelevant chunk pushed above two relevant ones: precision = (1/2 + 2/3) / 2.
    cp = context_precision(["x", "r1", "r2"], {"r1", "r2"})
    assert abs(cp - (0.5 + 2 / 3) / 2) < 1e-9


def test_ordering_affects_precision():
    top = context_precision(["r1", "x"], {"r1"})  # relevant first
    bottom = context_precision(["x", "r1"], {"r1"})  # relevant second
    assert top == 1.0
    assert bottom == 0.5
    assert top > bottom


# ----------------------------------------------------------------- context recall
def test_recall_fraction_of_relevant_retrieved():
    assert context_recall(["r1", "x"], {"r1", "r2"}) == 0.5
    assert context_recall(["r1", "r2", "x"], {"r1", "r2"}) == 1.0
    assert context_recall(["x"], {"r1"}) == 0.0


# ----------------------------------------------------------------- grounding / relevancy
def test_faithfulness_is_answer_terms_in_context():
    # Every content term of the answer appears in the context → 1.0.
    assert faithfulness_score("chest pain emergency", "new chest pain is an emergency") == 1.0
    # Half the answer's content terms missing → 0.5.
    assert faithfulness_score("chest pain emergency stroke", "chest pain emergency") < 1.0


def test_answer_relevancy_is_query_terms_in_answer():
    assert answer_relevancy("stroke signs", "stroke warning signs include drooping") == 1.0
    assert answer_relevancy("stroke fever signs", "stroke signs") < 1.0


# ----------------------------------------------------------------- real-corpus run
class _FixtureChunk:
    def __init__(self, chunk_id: str, text: str) -> None:
        self.chunk_id = chunk_id
        self.text = text


class _FixtureRetriever:
    """A minimal retriever returning a fixed ranking, to drive ``evaluate_retrieval`` units."""

    def __init__(self, ranking: list[_FixtureChunk]) -> None:
        self._ranking = ranking

    def retrieve(self, query: str, k: int = 5):
        return self._ranking[:k]


def test_evaluate_retrieval_aggregates_over_queries():
    from eval.retrieval_eval import RetrievalLabel

    retriever = _FixtureRetriever(
        [_FixtureChunk("r1", "chest pain emergency call 911"), _FixtureChunk("x", "unrelated")]
    )
    labels = [
        RetrievalLabel(query="chest pain emergency", relevant=["r1"], answer="chest pain emergency")
    ]
    rep = evaluate_retrieval(labels, retriever, k=5)
    assert rep.n_queries == 1
    assert rep.context_recall == 1.0
    assert rep.context_precision == 1.0
    assert rep.faithfulness == 1.0
    assert 0.0 <= rep.answer_relevancy <= 1.0


def test_evaluate_retrieval_over_real_corpus_is_in_range_and_relevant_surfaces():
    from scribeintake.config import settings
    from scribeintake.rag.embed import HashingEmbedder
    from scribeintake.rag.ingest import build_records
    from scribeintake.rag.rerank import LexicalReranker
    from scribeintake.rag.retrieve import build_retriever

    records = build_records(settings.KB_DIR)
    emb = HashingEmbedder()
    vecs = emb.embed_documents([r.text for r in records])
    retriever = build_retriever(
        records, embedder=emb, reranker=LexicalReranker(), dense_vectors=vecs
    )

    labels = load_retrieval_labels()
    rep = evaluate_retrieval(labels, retriever, k=5)

    # Every metric is a proper fraction.
    values = (rep.context_precision, rep.context_recall, rep.faithfulness, rep.answer_relevancy)
    for value in values:
        assert 0.0 <= value <= 1.0
    assert rep.n_queries == len(labels)
    # The label set was authored to be retrievable: recall is high and at least one clearly
    # relevant query lands its relevant chunk at the top (precision 1.0).
    assert rep.context_recall >= 0.8
    assert any(pq["context_precision"] == 1.0 for pq in rep.per_query)

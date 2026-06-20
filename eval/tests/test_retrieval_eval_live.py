"""Live RAGAS-style retrieval eval (``@pytest.mark.live``) — needs the real local models.

Builds a real index over the curated corpus with the live ``bge`` embedder + reranker and runs
``evaluate_retrieval`` over the held-out label set. **Skips** (not errors) if the heavy models
can't load — e.g. a broken local torch install — mirroring the Split-05 smoke test, so the live
suite degrades cleanly. (The metric math + a real-corpus run with the model-free hashing
embedder are already covered deterministically in ``test_retrieval_metrics.py``.)

Run: ``python -m pytest eval/tests/test_retrieval_eval_live.py -m live -v``
"""

from __future__ import annotations

import pytest

from eval.retrieval_eval import evaluate_retrieval, load_retrieval_labels

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live_retriever(tmp_path_factory):
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
    ingest(index_dir=index_dir, conn=conn, embedder=embedder)
    retriever = load_retriever(index_dir)
    yield retriever
    conn.close()


def test_retrieval_metrics_in_range_and_relevant_scores_high(live_retriever):
    labels = load_retrieval_labels()
    report = evaluate_retrieval(labels, live_retriever)

    print(
        f"\n[retrieval] n={report.n_queries} "
        f"ctx_precision={report.context_precision:.2f} ctx_recall={report.context_recall:.2f} "
        f"faithfulness={report.faithfulness:.2f} answer_relevancy={report.answer_relevancy:.2f}"
    )
    for value in (
        report.context_precision,
        report.context_recall,
        report.faithfulness,
        report.answer_relevancy,
    ):
        assert 0.0 <= value <= 1.0
    # A clearly-relevant query lands its relevant chunk near the top.
    assert report.context_recall >= 0.7
    assert any(pq["context_precision"] == 1.0 for pq in report.per_query)

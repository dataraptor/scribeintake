"""RAGAS-style retrieval metrics (Split 08, spec section 11) — a transparent local implementation.

Four metrics over a held-out **query → relevant-chunk** label set (``retrieval_labels.yaml``,
keyed by the stable Split-05 ``chunk_id``s):

* **context precision** — are the *retrieved* chunks relevant, and ranked well? (average precision
  over the retrieved-relevant chunks: a relevant chunk lower in the ranking scores less).
* **context recall** — of the labelled relevant chunks, how many were retrieved.
* **faithfulness** — is the reference answer grounded in the retrieved context? (fraction of the
  answer's content terms found in the retrieved chunk text).
* **answer relevancy** — does the reference answer address the query? (fraction of the query's
  content terms found in the answer).

**Local, not the ``ragas`` package** (recorded in :attr:`RetrievalReport.impl`): the math is a
handful of pure functions so every number is unit-testable on a fixture (spec section 11 — "don't
make the numbers a black box"). The content-term overlap reuses the same model-free tokeniser the
RAG path uses (:mod:`scribeintake.rag.text`), so faithfulness/relevancy score content words, not
stopwords.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean

import yaml
from pydantic import BaseModel, ConfigDict, Field

from scribeintake.config import RETRIEVE_K
from scribeintake.rag.text import overlap_score

from .models import RetrievalReport

DEFAULT_LABELS = Path(__file__).resolve().parent / "retrieval_labels.yaml"


class RetrievalLabel(BaseModel):
    """One held-out retrieval label: a query, its relevant ``chunk_id``s, and a reference answer."""

    model_config = ConfigDict(extra="forbid")

    query: str
    relevant: list[str] = Field(min_length=1)
    answer: str = ""
    note: str | None = None


def load_retrieval_labels(path: str | Path | None = None) -> list[RetrievalLabel]:
    """Load + validate the retrieval label set (defaults to ``eval/retrieval_labels.yaml``)."""
    p = Path(path) if path else DEFAULT_LABELS
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{p}: retrieval labels file must be a YAML list")
    return [RetrievalLabel.model_validate(item) for item in raw]


# ============================================================ pure metric math
def context_precision(retrieved_ids: list[str], relevant: set[str]) -> float:
    """Average precision over the retrieved-relevant chunks (ranking-aware).

    For each retrieved chunk that is relevant, take precision@(its rank) and average over the
    relevant chunks actually retrieved. All relevant chunks ranked first → 1.0; a relevant chunk
    buried below irrelevant ones scores less. 0.0 if nothing relevant was retrieved. Recall (the
    *missed* relevant chunks) is reported separately so the two signals stay orthogonal.
    """
    if not relevant:
        return 1.0
    hits = 0
    precision_sum = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / hits if hits else 0.0


def context_recall(retrieved_ids: list[str], relevant: set[str]) -> float:
    """Fraction of the labelled relevant chunks that were retrieved."""
    if not relevant:
        return 1.0
    return len(set(retrieved_ids) & relevant) / len(relevant)


def faithfulness_score(answer: str, context: str) -> float:
    """Fraction of the answer's content terms present in the retrieved context (grounding)."""
    return overlap_score(answer, context)


def answer_relevancy(query: str, answer: str) -> float:
    """Fraction of the query's content terms covered by the answer (does it address the query)."""
    return overlap_score(query, answer)


# ============================================================ end-to-end evaluation
def evaluate_retrieval(
    labels: list[RetrievalLabel], retriever: object, *, k: int = RETRIEVE_K
) -> RetrievalReport:
    """Run ``retriever`` over every label and aggregate the four metrics (mean over queries).

    ``retriever`` is any object exposing ``retrieve(query, k) -> [RetrievedChunk]`` (the live
    :class:`~scribeintake.rag.retrieve.HybridRetriever`, or a deterministic fixture). Per-query
    rows are kept on the report for transparency/debugging.
    """
    cps: list[float] = []
    crs: list[float] = []
    fas: list[float] = []
    ars: list[float] = []
    per_query: list[dict] = []

    for label in labels:
        retrieved = retriever.retrieve(label.query, k=k)  # type: ignore[attr-defined]
        retrieved_ids = [c.chunk_id for c in retrieved]
        context = " ".join(c.text for c in retrieved)
        relevant = set(label.relevant)

        cp = context_precision(retrieved_ids, relevant)
        cr = context_recall(retrieved_ids, relevant)
        fa = faithfulness_score(label.answer, context) if label.answer else None
        ar = answer_relevancy(label.query, label.answer) if label.answer else None

        cps.append(cp)
        crs.append(cr)
        if fa is not None:
            fas.append(fa)
        if ar is not None:
            ars.append(ar)
        per_query.append(
            {
                "query": label.query,
                "relevant": sorted(relevant),
                "retrieved": retrieved_ids,
                "context_precision": round(cp, 4),
                "context_recall": round(cr, 4),
                "faithfulness": round(fa, 4) if fa is not None else None,
                "answer_relevancy": round(ar, 4) if ar is not None else None,
            }
        )

    return RetrievalReport(
        context_precision=mean(cps) if cps else 0.0,
        context_recall=mean(crs) if crs else 0.0,
        faithfulness=mean(fas) if fas else 0.0,
        answer_relevancy=mean(ars) if ars else 0.0,
        n_queries=len(labels),
        k=k,
        per_query=per_query,
    )

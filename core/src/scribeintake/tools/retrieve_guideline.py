"""``retrieve_guideline`` tool (spec §8/§11) — real hybrid retrieval over the local KB.

Split 05 swaps the Split-03 stub for the production retriever: it calls the injected
:class:`~scribeintake.rag.HybridRetriever` (BM25 ∪ dense → cross-encoder rerank → top-k) and
returns real ``chunk_id``-bearing chunks. **Local tool — no model call, cost $0** in the trace.

Retrieval is always best-effort: with no retriever wired (``ctx.retriever is None``, e.g. the
deterministic tier or an unbuilt index) or on any retrieval error, it returns ``{"chunks": []}``
— treated downstream as "no citation available", never a fabricated source (spec §18).
"""

from __future__ import annotations

import logging

from ..config import RETRIEVE_K
from ..models import RetrieveGuidelineInput
from .base import ToolContext, ToolSpec

logger = logging.getLogger(__name__)

_DESCRIPTION = (
    "Retrieve passages from curated, public-domain clinical guidelines (MedlinePlus, CDC, NIH, "
    "NIMH) to ground a statement. Returns cited chunks, each with a chunk_id, source and url; "
    "never invent a source. An empty result means no citation is available."
)


def execute(arguments: dict, ctx: ToolContext) -> dict:
    """Run hybrid retrieval for ``arguments['query']``; return ``{"chunks": [...]}``."""
    args = RetrieveGuidelineInput.model_validate(arguments)  # shape-check; raises on bad args
    retriever = ctx.retriever
    if retriever is None:
        return {"chunks": []}  # no index wired → graceful "no citation available"
    try:
        chunks = retriever.retrieve(args.query, k=args.k or RETRIEVE_K)
    except Exception as exc:  # noqa: BLE001 - retrieval must never crash the turn
        logger.warning("retrieve_guideline failed, returning no chunks: %s", exc)
        return {"chunks": []}
    return {"chunks": [c.model_dump() for c in chunks]}


SPEC = ToolSpec(
    name="retrieve_guideline",
    description=_DESCRIPTION,
    parameters=RetrieveGuidelineInput.model_json_schema(),
    executor=execute,
)

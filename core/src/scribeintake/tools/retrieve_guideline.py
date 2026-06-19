"""``retrieve_guideline`` tool — **STUB** for Split 03 (spec section 8 / 11).

Returns an empty result this split; Split 05 swaps the body in (BM25 + embeddings + rerank
over the local KB) without changing the signature or the callers. The return shape is fixed
now: ``{"chunks": list[RetrievedChunk]}``. Local tool — **no model call** (cost ``$0``).
"""

from __future__ import annotations

from ..models import RetrieveGuidelineInput
from .base import ToolContext, ToolSpec

_DESCRIPTION = (
    "Retrieve passages from curated, public-domain clinical guidelines to ground a "
    "statement. Returns cited chunks; never invent a source. (No results yet — retrieval "
    "is wired in a later build; treat an empty result as 'no citation available'.)"
)


def execute(arguments: dict, ctx: ToolContext) -> dict:
    """Validate the query shape and return no chunks (stub)."""
    RetrieveGuidelineInput.model_validate(arguments)  # shape-check; raises on bad args
    return {"chunks": []}


SPEC = ToolSpec(
    name="retrieve_guideline",
    description=_DESCRIPTION,
    parameters=RetrieveGuidelineInput.model_json_schema(),
    executor=execute,
)

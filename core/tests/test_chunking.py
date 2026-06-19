"""Chunking (Split 05 §4) — section-aware, sized, and stable. Deterministic (no models).

Covers both a controlled synthetic doc and the real curated corpus: chunks respect the ~400
token cap, are never empty, and ``chunk_id``s are stable + unique across two runs over the same
input — the property eval/citation references depend on.
"""

from __future__ import annotations

from scribeintake.config import settings
from scribeintake.rag.chunking import (
    TARGET_MAX_TOKENS,
    chunk_text,
    estimate_tokens,
    make_chunk_id,
)
from scribeintake.rag.ingest import build_records

_DOC = """# Title

## Section One
Alpha bravo charlie delta echo foxtrot. Golf hotel india juliet kilo lima mike november.
Oscar papa quebec romeo sierra tango. Uniform victor whiskey xray yankee zulu one two three.

## Section Two
Repeat one two three four five six seven eight nine ten eleven twelve thirteen fourteen.
Fifteen sixteen seventeen eighteen nineteen twenty twentyone twentytwo twentythree.
"""


def test_chunks_are_nonempty_and_within_cap():
    chunks = chunk_text(_DOC)
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert c.text.strip(), "no chunk may be empty"
        assert estimate_tokens(c.text) <= TARGET_MAX_TOKENS


def test_section_heading_is_carried():
    chunks = chunk_text(_DOC)
    sections = {c.section for c in chunks}
    # The starting heading of each chunk is recorded (small doc → one packed chunk here).
    assert sections <= {"Section One", "Section Two"}


def test_oversized_paragraph_is_split_by_sentences():
    big = "# T\n\n" + " ".join(f"Word{i} alpha beta gamma delta." for i in range(400))
    chunks = chunk_text(big)
    assert len(chunks) >= 2  # a single huge paragraph is broken up
    for c in chunks:
        assert estimate_tokens(c.text) <= TARGET_MAX_TOKENS


def test_chunk_id_is_stable_and_deterministic():
    assert make_chunk_id("heart_attack.md", 0) == make_chunk_id("heart_attack.md", 0)
    # Different doc OR different index → different id (no cross-doc collision on shared source).
    assert make_chunk_id("heart_attack.md", 0) != make_chunk_id("chest_pain.md", 0)
    assert make_chunk_id("heart_attack.md", 0) != make_chunk_id("heart_attack.md", 1)
    assert make_chunk_id("heart_attack.md", 0).startswith("chk_")


# ----------------------------------------------------------------- real corpus
def test_real_corpus_chunks_are_sized_and_unique():
    records = build_records(settings.KB_DIR)
    assert 18 <= len(records) <= 40  # ~20-25 chunked pages (acceptance #1)
    sizes = [estimate_tokens(r.text) for r in records]
    assert max(sizes) <= TARGET_MAX_TOKENS
    assert min(sizes) >= 60  # no slivers
    # Most chunks land in the ~200-400 token band.
    assert sum(1 for s in sizes if 180 <= s <= TARGET_MAX_TOKENS) >= len(records) // 2
    ids = [r.chunk_id for r in records]
    assert len(set(ids)) == len(ids), "chunk_ids must be unique"
    # every chunk carries provenance
    for r in records:
        assert r.source and r.url and r.license


def test_real_corpus_chunk_ids_stable_across_runs():
    a = [r.chunk_id for r in build_records(settings.KB_DIR)]
    b = [r.chunk_id for r in build_records(settings.KB_DIR)]
    assert a == b

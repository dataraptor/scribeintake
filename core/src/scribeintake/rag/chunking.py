"""Section-aware document chunking + stable chunk ids (spec §11) — **no models**.

Pure text logic, so it is covered in the deterministic test tier without any model download.
Docs are Markdown: ``#``/``##``/``###`` headings delimit *sections*; each section is split
into ~200–400-token chunks at paragraph boundaries (never mid-sentence unless a single
paragraph is itself oversized). A token is approximated as ``words / 0.75`` (≈1.33 tokens per
whitespace word) — close enough to size chunks without pulling in a tokenizer dependency.

``chunk_id`` is a stable hash of ``doc|chunk-index`` (the doc filename is unique, so ids never
collide across same-source pages, and the locator — not the prose — is hashed, so editing
wording keeps ids that eval scenarios (Split 06) and faithfulness scoring (Split 08) reference).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# Word→token approximation: English averages ~0.75 words per token.
_WORDS_PER_TOKEN = 0.75
TARGET_MAX_TOKENS = 400
TARGET_MIN_TOKENS = 120  # below this a trailing chunk is merged back if possible

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass(frozen=True)
class Chunk:
    """One chunk: its section heading, its index within that section, and its text."""

    section: str
    index: int
    text: str


def estimate_tokens(text: str) -> int:
    """Approximate token count of ``text`` (``words / 0.75``). Deterministic, no tokenizer."""
    words = len(re.findall(r"\S+", text))
    return round(words / _WORDS_PER_TOKEN)


def make_chunk_id(doc: str, index: int) -> str:
    """Stable, deterministic chunk id: ``chk_<10 hex>`` of ``doc|index``.

    Keyed by the unique doc filename + its chunk index, so ids are unique across same-source
    pages and stable across wording edits (the locator, not the prose, is hashed) — what
    downstream eval/citation references depend on.
    """
    key = f"{doc}␟{index}".encode()
    return "chk_" + hashlib.sha1(key).hexdigest()[:10]


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown into ``(section_heading, body)`` pairs (preamble → 'Overview')."""
    sections: list[tuple[str, list[str]]] = []
    current = "Overview"
    buf: list[str] = []
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            if buf:
                sections.append((current, buf))
            current = m.group(2).strip()
            buf = []
        else:
            buf.append(line)
    if buf:
        sections.append((current, buf))
    return [(name, "\n".join(lines).strip()) for name, lines in sections if "".join(lines).strip()]


def _split_paragraphs(body: str) -> list[str]:
    """Blank-line-delimited paragraphs, whitespace-normalised, empties dropped."""
    paras = re.split(r"\n\s*\n", body)
    return [re.sub(r"\s+", " ", p).strip() for p in paras if p.strip()]


def _split_oversized(paragraph: str) -> list[str]:
    """Break a paragraph longer than the max into sentence-grouped pieces under the cap."""
    sentences = _SENTENCE_RE.split(paragraph)
    pieces: list[str] = []
    cur: list[str] = []
    for sent in sentences:
        cur.append(sent)
        if estimate_tokens(" ".join(cur)) >= TARGET_MAX_TOKENS:
            pieces.append(" ".join(cur).strip())
            cur = []
    if cur:
        pieces.append(" ".join(cur).strip())
    return pieces or [paragraph]


def chunk_text(text: str) -> list[Chunk]:
    """Section-aware chunking: ~200–400-token chunks that respect section boundaries.

    Paragraphs are walked in document order (carrying their section heading) and greedily
    packed — across small adjacent sections — until adding the next paragraph would exceed
    :data:`TARGET_MAX_TOKENS`; an oversized single paragraph is split by sentences. A chunk's
    ``section`` is the heading where it begins. A tiny trailing chunk (< :data:`TARGET_MIN_TOKENS`)
    is merged back when the result still fits the cap, so we don't emit slivers.
    """
    blocks: list[tuple[str, str]] = []  # (section, paragraph) in document order
    for section, body in _split_sections(text):
        for para in _split_paragraphs(body):
            if estimate_tokens(para) > TARGET_MAX_TOKENS:
                blocks.extend((section, piece) for piece in _split_oversized(para))
            else:
                blocks.append((section, para))

    packed: list[tuple[str, str]] = []  # (section, chunk_text)
    cur: list[str] = []
    cur_section = ""
    for section, para in blocks:
        if cur and estimate_tokens(" ".join([*cur, para])) > TARGET_MAX_TOKENS:
            packed.append((cur_section, " ".join(cur)))
            cur, cur_section = [para], section
        else:
            if not cur:
                cur_section = section
            cur.append(para)
    if cur:
        packed.append((cur_section, " ".join(cur)))

    packed = _merge_slivers(packed)
    return [Chunk(section=sec, index=i, text=txt.strip()) for i, (sec, txt) in enumerate(packed)]


def _merge_slivers(packed: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Fold an undersized trailing chunk into its predecessor when the merge still fits."""
    if len(packed) >= 2 and estimate_tokens(packed[-1][1]) < TARGET_MIN_TOKENS:
        prev_sec, prev_txt = packed[-2]
        merged = prev_txt + " " + packed[-1][1]
        if estimate_tokens(merged) <= TARGET_MAX_TOKENS:
            return [*packed[:-2], (prev_sec, merged)]
    return packed

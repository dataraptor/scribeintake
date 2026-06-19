"""Corpus ingestion (spec §11) — chunk → local embed → persist (vector store + BM25 + kb_chunks).

Reads the curated public-domain pages in ``kb/docs/`` plus their provenance in
``kb/sources.csv``, section-aware chunks each page, embeds every chunk with the **local**
``bge-small-en-v1.5`` model, and persists to three stores under ``config.RAG_INDEX_DIR``:

* **``vectors.json``** — the dense vector store (chunk records + their embeddings). A
  dependency-light local store standing in for the spec's Chroma pin (see the Split 05 session
  log: chromadb 0.4.24 ⊥ NumPy 2.x); the ``DenseIndex`` seam keeps Chroma/FAISS a drop-in.
* **``bm25.json``** — the tokenised corpus (the persisted sparse store).
* the **``kb_chunks``** SQLite table — the queryable/audit copy with full provenance.

Idempotent: a re-run rebuilds all three from source (dev recreates from synthetic/curated
data). ``chunk_id``s are stable across re-ingests (hash of the doc filename + chunk index), so
citations stay reproducible. Run it with ``python -m scribeintake.rag.ingest`` (or ``make
ingest`` / ``.\\tasks.ps1 ingest``).
"""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import settings
from .chunking import chunk_text, make_chunk_id
from .retrieve import VECTORS_STORE, ChunkRecord
from .text import tokenize

logger = logging.getLogger(__name__)

SOURCES_CSV = "sources.csv"
DOCS_DIR = "docs"
BM25_STORE = "bm25.json"

# sources.csv columns (spec §13): the `doc` column names the file in kb/docs/.
_CSV_FIELDS = ("doc", "source", "url", "license", "jurisdiction", "last_reviewed", "section")


@dataclass(frozen=True)
class SourceRow:
    """One row of ``kb/sources.csv`` — a curated page and its provenance."""

    doc: str
    source: str
    url: str
    license: str
    jurisdiction: str
    last_reviewed: str
    section: str = ""


def load_sources(kb_dir: Path) -> list[SourceRow]:
    """Parse ``kb/sources.csv`` into :class:`SourceRow` records."""
    path = kb_dir / SOURCES_CSV
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [SourceRow(**{k: (r.get(k) or "").strip() for k in _CSV_FIELDS}) for r in reader]
    if not rows:
        raise ValueError(f"no sources found in {path}")
    return rows


def build_records(kb_dir: Path) -> list[ChunkRecord]:
    """Chunk every curated doc into provenance-carrying :class:`ChunkRecord`s (deterministic)."""
    records: list[ChunkRecord] = []
    for row in load_sources(kb_dir):
        doc_path = kb_dir / DOCS_DIR / row.doc
        text = doc_path.read_text(encoding="utf-8")
        for chunk in chunk_text(text):
            records.append(
                ChunkRecord(
                    chunk_id=make_chunk_id(row.doc, chunk.index),
                    text=chunk.text,
                    source=row.source,
                    url=row.url,
                    license=row.license,
                    jurisdiction=row.jurisdiction,
                    section=chunk.section,
                    last_reviewed=row.last_reviewed,
                )
            )
    # Stable order (helps reproducibility) and de-dup on chunk_id (defensive).
    seen: dict[str, ChunkRecord] = {}
    for r in sorted(records, key=lambda r: r.chunk_id):
        seen.setdefault(r.chunk_id, r)
    return list(seen.values())


def _persist_vectors(
    records: list[ChunkRecord],
    embeddings: list[list[float]],
    embed_model: str,
    index_dir: Path,
) -> None:
    """Persist the dense vector store: chunk records + their embeddings (parallel lists).

    A dependency-light JSON store (the Chroma stand-in). ``embed_model`` records which embedder
    produced the vectors so a loader can use a matching query encoder. Idempotent: overwrites.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    store = {
        "embed_model": embed_model,
        "records": [asdict(r) for r in records],
        "embeddings": embeddings,
    }
    (index_dir / VECTORS_STORE).write_text(json.dumps(store), encoding="utf-8")


def _persist_bm25(records: list[ChunkRecord], index_dir: Path) -> None:
    """Persist the tokenised corpus alongside the vector store (the sparse store)."""
    store = {
        "chunk_ids": [r.chunk_id for r in records],
        "tokens": [tokenize(r.text) for r in records],
    }
    (index_dir / BM25_STORE).write_text(json.dumps(store), encoding="utf-8")


def _persist_kb_chunks(records: list[ChunkRecord], conn: sqlite3.Connection) -> None:
    """Replace the ``kb_chunks`` table contents with the freshly-ingested chunks."""
    conn.execute("DELETE FROM kb_chunks")
    conn.executemany(
        "INSERT INTO kb_chunks "
        "(id, source, url, license, jurisdiction, section, last_reviewed, text) "
        "VALUES (:chunk_id, :source, :url, :license, :jurisdiction, :section, "
        ":last_reviewed, :text)",
        [{**asdict(r)} for r in records],
    )
    conn.commit()


def ingest(
    kb_dir: Path | None = None,
    index_dir: Path | None = None,
    conn: sqlite3.Connection | None = None,
    *,
    embedder: object | None = None,
) -> int:
    """Build the corpus index end-to-end; returns the number of chunks ingested.

    ``embedder`` defaults to the live local ``bge-small-en-v1.5``; tests may inject the
    deterministic :class:`~scribeintake.rag.embed.HashingEmbedder`. A fresh SQLite connection is
    opened (and initialised) if one is not supplied.
    """
    kb_dir = Path(kb_dir) if kb_dir is not None else settings.KB_DIR
    index_dir = Path(index_dir) if index_dir is not None else settings.RAG_INDEX_DIR

    records = build_records(kb_dir)
    if not records:
        raise ValueError(f"no chunks produced from {kb_dir / DOCS_DIR}")

    if embedder is None:
        from .embed import load_default_embedder

        embedder = load_default_embedder()
    embeddings = embedder.embed_documents([r.text for r in records])  # type: ignore[attr-defined]
    embed_model = getattr(embedder, "model_name", type(embedder).__name__)

    own_conn = conn is None
    if own_conn:
        from .. import db

        conn = db.connect()
        db.init_db(conn)

    try:
        _persist_vectors(records, embeddings, embed_model, index_dir)
        _persist_bm25(records, index_dir)
        _persist_kb_chunks(records, conn)
    finally:
        if own_conn:
            conn.close()

    logger.info("ingested %d chunks from %s", len(records), kb_dir)
    return len(records)


def main() -> None:
    """CLI entry: ``python -m scribeintake.rag.ingest``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n = ingest()
    print(f"Ingested {n} chunks → {settings.RAG_INDEX_DIR} (vectors.json + bm25.json) + kb_chunks.")


if __name__ == "__main__":
    main()

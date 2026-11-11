"""Document ingestion: load -> chunk -> embed -> index -> persist.

Chunking is structure-aware: Markdown headers split documents into sections first (the
header path is kept as metadata and prepended to each chunk, so a chunk about "penalties"
still knows it belongs to the Supplier SLA), then a recursive character splitter cuts long
sections into overlapping chunks.

Embedding concurrency depends on where the work happens:

* local hashing embeddings are CPU-bound pure Python -> large corpora go to a
  ``ProcessPoolExecutor`` (threads would serialise on the GIL); small ones run inline
  because process start-up costs more than the work (measured, see profile_ingest.py);
* API embeddings (OpenAI / Gemini) are I/O-bound -> batches run concurrently on the event
  loop with ``asyncio.gather`` and a semaphore that caps in-flight requests.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from retail_agent.config import Settings
from retail_agent.llm import embedding_signature, get_embeddings
from retail_agent.rag.hashing import hash_embed_many, hash_embed_parallel
from retail_agent.rag.store import Chunk, FaissStore

log = logging.getLogger(__name__)

_HEADERS = [("#", "title"), ("##", "section"), ("###", "subsection")]


@dataclass
class IngestStats:
    documents: int
    chunks: int
    duplicates_skipped: int
    chunk_ms: float
    embed_ms: float
    total_ms: float
    embedding: str


def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    files = sorted(p for p in docs_dir.glob("*.md") if p.is_file())
    return [(p.name, p.read_text(encoding="utf-8")) for p in files]


def strip_boilerplate(docs: list[tuple[str, str]], min_share: float = 0.5) -> list[tuple[str, str]]:
    """Drop lines repeated in at least ``min_share`` of documents (disclaimers, footers).

    Repeated boilerplate otherwise becomes near-identical chunks that match many queries
    and push useful context out of the top-k.
    """
    if len(docs) < 3:
        return docs
    line_df: dict[str, int] = {}
    for _, text in docs:
        for line in {ln.strip() for ln in text.splitlines() if len(ln.strip()) > 30}:
            line_df[line] = line_df.get(line, 0) + 1
    repeated = {ln for ln, n in line_df.items() if n / len(docs) >= min_share}
    if not repeated:
        return docs
    return [
        (name, "\n".join(ln for ln in text.splitlines() if ln.strip() not in repeated))
        for name, text in docs
    ]


def chunk_documents(
    docs: list[tuple[str, str]], chunk_size: int, chunk_overlap: int
) -> tuple[list[Chunk], int]:
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS, strip_headers=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    seen: set[str] = set()
    duplicates = 0
    for source, text in docs:
        position = 0
        for section_doc in header_splitter.split_text(text):
            md = section_doc.metadata
            title = md.get("title", source)
            section_path = " > ".join(v for k in ("section", "subsection") if (v := md.get(k)))
            for piece in splitter.split_text(section_doc.page_content):
                body = piece.strip()
                if len(body) < 20:  # skip fragments such as a lone italic disclaimer
                    continue
                digest = hashlib.sha1(body.encode("utf-8")).hexdigest()
                if digest in seen:  # exact duplicate text anywhere in the corpus
                    duplicates += 1
                    continue
                seen.add(digest)
                header = f"{title} | {section_path}" if section_path else title
                chunks.append(Chunk(
                    chunk_id=hashlib.sha1(f"{source}:{position}:{digest}".encode()).hexdigest()[:12],
                    text=f"[{header}]\n{body}",
                    source=source,
                    section=section_path or title,
                    position=position,
                ))
                position += 1
    return chunks, duplicates


def embed_local_parallel(
    texts: list[str], dim: int, workers: int, batch_size: int = 1024, min_parallel: int = 5000
) -> np.ndarray:
    """CPU-bound embedding; uses worker processes only when it pays off.

    Measured on an 8-core M2 (scripts/profile_ingest.py): starting 4 workers costs about
    0.2 s, so the pool loses below ~5,000 chunks and is ~2.7x faster at 30k-100k chunks.
    """
    if workers <= 1 or len(texts) < min_parallel:
        return hash_embed_many(texts, dim)
    return hash_embed_parallel(texts, dim, workers, batch_size)


async def embed_api_concurrent(
    embeddings: Embeddings, texts: list[str], batch_size: int = 96, max_in_flight: int = 4
) -> np.ndarray:
    """I/O-bound embedding: concurrent API calls, capped by a semaphore."""
    sem = asyncio.Semaphore(max_in_flight)

    async def one(batch: list[str]) -> list[list[float]]:
        async with sem:
            return await embeddings.aembed_documents(batch)

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    parts = await asyncio.gather(*(one(b) for b in batches))  # order is preserved
    return np.asarray([v for part in parts for v in part], dtype=np.float32)


async def build_index(settings: Settings, save: bool = True) -> tuple[FaissStore, IngestStats]:
    t0 = time.perf_counter()
    docs = load_documents(settings.docs_dir)
    if not docs:
        raise FileNotFoundError(f"No .md documents found in {settings.docs_dir}")
    docs = strip_boilerplate(docs)
    chunks, dupes = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    t1 = time.perf_counter()

    texts = [c.text for c in chunks]
    if settings.embedding_provider == "local":
        vectors = await asyncio.to_thread(
            embed_local_parallel, texts, settings.local_embedding_dim, settings.embed_workers,
            min_parallel=settings.parallel_embed_min_chunks,
        )
    else:
        vectors = await embed_api_concurrent(get_embeddings(settings), texts)
    t2 = time.perf_counter()

    store = FaissStore(dim=vectors.shape[1])
    store.add(chunks, vectors)
    sig = embedding_signature(settings)
    if save:
        store.save(settings.index_dir, {
            "embedding": sig,
            "dim": int(vectors.shape[1]),
            "chunks": len(chunks),
            "documents": [name for name, _ in docs],
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        })
    stats = IngestStats(
        documents=len(docs),
        chunks=len(chunks),
        duplicates_skipped=dupes,
        chunk_ms=round((t1 - t0) * 1000, 1),
        embed_ms=round((t2 - t1) * 1000, 1),
        total_ms=round((time.perf_counter() - t0) * 1000, 1),
        embedding=sig,
    )
    log.info("index built", extra={"fields": stats.__dict__})
    return store, stats

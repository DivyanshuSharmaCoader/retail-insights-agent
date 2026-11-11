from __future__ import annotations

import numpy as np
import pytest

from retail_agent.llm import HashingEmbeddings
from retail_agent.rag.ingest import (
    chunk_documents,
    embed_local_parallel,
    load_documents,
    strip_boilerplate,
)
from retail_agent.rag.retriever import IndexMismatchError, Retriever
from retail_agent.rag.store import Chunk, FaissStore, mmr_select
from scripts.eval_retrieval import evaluate, load_golden


def test_boilerplate_lines_are_removed() -> None:
    footer = "This line is a shared legal disclaimer repeated everywhere."
    docs = [(f"d{i}.md", f"# Doc {i}\n{footer}\nUnique content number {i}.") for i in range(4)]
    cleaned = strip_boilerplate(docs)
    assert all(footer not in text for _, text in cleaned)
    assert all(f"Unique content number {i}" in text for i, (_, text) in enumerate(cleaned))


def test_chunks_carry_section_metadata(settings) -> None:
    chunks, _ = chunk_documents(load_documents(settings.docs_dir), 700, 100)
    penalties = [c for c in chunks if c.source == "supplier_sla.md" and c.section == "Penalties"]
    assert penalties, "header-aware splitting should keep the 'Penalties' section"
    assert penalties[0].text.startswith("[Kestrel Retail Supplier Service Level Agreement")
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_mmr_skips_near_duplicates() -> None:
    a = np.array([1.0, 0.0, 0.0])
    near_dup = np.array([0.99, 0.01, 0.0])
    different = np.array([0.6, 0.8, 0.0])
    vecs = np.vstack([a, near_dup, different]).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    picked = mmr_select([1.0, 0.99, 0.7], vecs, k=2, lambda_mult=0.5)
    assert picked == [0, 2]


def test_parallel_embedding_matches_sequential() -> None:
    texts = [f"store {i} sold {i * 3} units of dairy and snacks" for i in range(300)]
    sequential = np.asarray(HashingEmbeddings(256).embed_documents(texts), dtype=np.float32)
    parallel = embed_local_parallel(texts, dim=256, workers=2, batch_size=50, min_parallel=0)
    assert np.allclose(sequential, parallel)


def test_store_save_load_roundtrip(tmp_path) -> None:
    emb = HashingEmbeddings(64)
    chunks = [Chunk(f"c{i}", f"text about topic {i}", "doc.md", "s", i) for i in range(5)]
    store = FaissStore(64)
    store.add(chunks, np.asarray(emb.embed_documents([c.text for c in chunks])))
    store.save(tmp_path, {"embedding": "x"})
    loaded, manifest = FaissStore.load(tmp_path)
    assert len(loaded) == 5 and manifest["embedding"] == "x"
    assert loaded.search(np.asarray(emb.embed_query("topic 3")), 1)[0][0] == 3


async def test_hybrid_retrieval_finds_the_right_section(retriever) -> None:
    hits = await retriever.aretrieve("What penalty does a supplier pay if fill rate is low?")
    assert hits[0].chunk.source == "supplier_sla.md"
    assert any(h.chunk.section == "Penalties" for h in hits)


async def test_off_topic_questions_retrieve_nothing(retriever) -> None:
    assert await retriever.aretrieve("What is the capital of France?") == []


async def test_golden_set_regression(retriever) -> None:
    report = await evaluate(retriever, load_golden(), k=4)
    assert report["hit@4"] >= 0.9
    assert report["off_topic_rejected"] == "3/3"


def test_index_built_with_other_embeddings_is_rejected(settings) -> None:
    other = settings.model_copy(update={"local_embedding_dim": 512})
    with pytest.raises(IndexMismatchError):
        Retriever.from_settings(other)

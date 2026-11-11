"""Query-time retrieval: hybrid (dense + BM25) candidates, rank fusion, relevance gate, MMR.

Pipeline for one question:

1. Dense: embed the query, take the ``fetch_k`` nearest chunks from FAISS (cosine).
2. Sparse: take the ``fetch_k`` best BM25 chunks.
3. Fuse both rankings with Reciprocal Rank Fusion (RRF): ``sum(1 / (60 + rank))``. RRF
   needs no score calibration between the two very different scales.
4. Relevance gate: keep a candidate only if ``cos/gate_dense + bm25/gate_bm25 >= 1``
   (either signal alone at its floor, or a mix of both). Off-topic questions then
   retrieve nothing, so the agent can say "not in the documents" instead of answering
   from weak context.
5. MMR over the survivors so the final ``k`` chunks are relevant *and* not redundant.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

import numpy as np
from langchain_core.embeddings import Embeddings

from retail_agent.config import Settings
from retail_agent.llm import HashingEmbeddings, embedding_signature, get_embeddings
from retail_agent.rag.bm25 import BM25Index
from retail_agent.rag.store import FaissStore, ScoredChunk, mmr_select

RRF_K = 60


class IndexMismatchError(RuntimeError):
    pass


class Retriever:
    def __init__(
        self,
        store: FaissStore,
        embeddings: Embeddings,
        k: int = 4,
        fetch_k: int = 12,
        lambda_mult: float = 0.6,
        gate_dense: float = 0.20,
        gate_bm25: float = 6.0,
        dense_floor: float = 0.12,
        hybrid: bool = True,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.gate_dense = gate_dense
        self.gate_bm25 = gate_bm25
        self.dense_floor = dense_floor
        self.hybrid = hybrid
        self.bm25 = BM25Index([c.text for c in store.chunks])
        if isinstance(embeddings, HashingEmbeddings):
            # Pure function of the text, so repeated questions can skip the work.
            self._embed_local = lru_cache(maxsize=1024)(
                lambda q: tuple(embeddings.embed_query(q))
            )

    @classmethod
    def from_settings(cls, settings: Settings, store: FaissStore | None = None) -> Retriever:
        if store is None:
            store, manifest = FaissStore.load(settings.index_dir)
            expected = embedding_signature(settings)
            if manifest.get("embedding") != expected:
                raise IndexMismatchError(
                    f"Index was built with '{manifest.get('embedding')}' but the current "
                    f"embedding setting is '{expected}'. Rebuild the index."
                )
        return cls(
            store,
            get_embeddings(settings),
            k=settings.retrieval_k,
            fetch_k=settings.retrieval_fetch_k,
            lambda_mult=settings.mmr_lambda,
            gate_dense=settings.gate_dense,
            gate_bm25=settings.gate_bm25,
            dense_floor=settings.dense_floor,
            hybrid=settings.hybrid_search,
        )

    async def _embed_query(self, query: str) -> np.ndarray:
        if isinstance(self.embeddings, HashingEmbeddings):
            vec = await asyncio.to_thread(self._embed_local, query)
        else:
            vec = await self.embeddings.aembed_query(query)
        return np.asarray(vec, dtype=np.float32)

    def _passes_gate(self, cos: float, bm25: float) -> bool:
        if not self.hybrid:
            return cos >= self.dense_floor
        return cos / self.gate_dense + bm25 / self.gate_bm25 >= 1.0

    async def aretrieve(self, query: str, k: int | None = None) -> list[ScoredChunk]:
        k = k or self.k
        qvec = await self._embed_query(query)
        dense = self.store.search(qvec, self.fetch_k)
        sparse = self.bm25.top(query, self.fetch_k) if self.hybrid else []

        fused: dict[int, float] = {}
        for rank, (pos, _) in enumerate(dense):
            fused[pos] = fused.get(pos, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (pos, _) in enumerate(sparse):
            fused[pos] = fused.get(pos, 0.0) + 1.0 / (RRF_K + rank + 1)

        positions = sorted(fused, key=fused.get, reverse=True)
        cos = dict(zip(positions, self.store.cosine(qvec, positions), strict=True))
        bm = {p: s for p, s in sparse}
        kept = [p for p in positions if self._passes_gate(cos[p], bm.get(p, 0.0))]
        if not kept:
            return []

        top = max(fused[p] for p in kept)
        relevance = [fused[p] / top for p in kept]
        vecs = np.vstack([self.store.vector(p) for p in kept])
        picked = mmr_select(relevance, vecs, k, self.lambda_mult)
        return [
            ScoredChunk(self.store.chunks[kept[i]], round(relevance[i], 4),
                        dense=round(cos[kept[i]], 4), sparse=round(bm.get(kept[i], 0.0), 3))
            for i in picked
        ]

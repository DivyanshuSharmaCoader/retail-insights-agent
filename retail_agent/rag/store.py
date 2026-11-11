"""A small FAISS vector store plus Maximal Marginal Relevance (MMR) selection.

Vectors are L2-normalised and stored in ``IndexFlatIP``, so inner product = cosine
similarity and the search is exact. For a few thousand chunks exact search takes well
under a millisecond; IVF or HNSW indexes only pay off at hundreds of thousands of vectors.

Persistence uses ``faiss.write_index`` plus JSON for chunk text and metadata (no pickle,
so loading an index never executes code).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import faiss
import numpy as np


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    section: str
    position: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float  # fused relevance in [0, 1] (1 = best candidate for this query)
    dense: float = 0.0  # cosine similarity to the query
    sparse: float = 0.0  # BM25 score


def mmr_select(
    relevance: list[float], vectors: np.ndarray, k: int, lambda_mult: float
) -> list[int]:
    """Greedy Maximal Marginal Relevance over candidate positions.

    Repeatedly picks the candidate maximising
    ``lambda * relevance(c) - (1 - lambda) * max(sim(c, already_selected))``,
    so near-duplicate chunks do not fill every slot in the context window.
    """
    if len(relevance) == 0:
        return []
    pairwise = vectors @ vectors.T
    selected: list[int] = []
    remaining = list(range(len(relevance)))
    while remaining and len(selected) < k:
        best_pos, best_val = remaining[0], -np.inf
        for pos in remaining:
            redundancy = max((float(pairwise[pos, s]) for s in selected), default=0.0)
            value = lambda_mult * relevance[pos] - (1 - lambda_mult) * redundancy
            if value > best_val:
                best_pos, best_val = pos, value
        selected.append(best_pos)
        remaining.remove(best_pos)
    return selected


class FaissStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    @staticmethod
    def normalise(vectors: np.ndarray) -> np.ndarray:
        vectors = np.array(vectors, dtype=np.float32, copy=True, ndmin=2)
        faiss.normalize_L2(vectors)
        return vectors

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        self.index.add(self.normalise(vectors))
        self.chunks.extend(chunks)

    def search(self, query: np.ndarray, n: int) -> list[tuple[int, float]]:
        """Exact nearest neighbours: (chunk position, cosine similarity), best first."""
        if not self.chunks:
            return []
        scores, ids = self.index.search(self.normalise(query), min(n, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i != -1]

    def cosine(self, query: np.ndarray, positions: list[int]) -> list[float]:
        q = self.normalise(query)[0]
        return [float(self.vector(p) @ q) for p in positions]

    def vector(self, position: int) -> np.ndarray:
        return self.index.reconstruct(int(position))

    def mmr_search(
        self, query: np.ndarray, k: int = 4, fetch_k: int = 12, lambda_mult: float = 0.6
    ) -> list[ScoredChunk]:
        """Dense-only MMR: fetch ``fetch_k`` neighbours, then pick ``k`` diverse ones."""
        hits = self.search(query, fetch_k)
        if not hits:
            return []
        vecs = np.vstack([self.vector(p) for p, _ in hits])
        picked = mmr_select([s for _, s in hits], vecs, k, lambda_mult)
        return [ScoredChunk(self.chunks[hits[i][0]], hits[i][1], dense=hits[i][1]) for i in picked]

    def save(self, directory: Path, manifest: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> tuple[FaissStore, dict]:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        index = faiss.read_index(str(directory / "index.faiss"))
        store = cls(index.d)
        store.index = index
        with (directory / "chunks.jsonl").open(encoding="utf-8") as fh:
            store.chunks = [Chunk(**json.loads(line)) for line in fh if line.strip()]
        if len(store.chunks) != index.ntotal:
            raise ValueError("index and chunk file are out of sync; rebuild the index")
        return store, manifest

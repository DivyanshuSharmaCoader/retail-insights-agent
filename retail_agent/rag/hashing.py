"""Hashing-trick embeddings as plain NumPy, kept import-light for worker processes.

Profiling showed that most of a worker's start-up time was spent importing LangChain, so
the CPU-bound code lives here with only NumPy and the tokeniser as dependencies.
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from retail_agent.text import tokenize


def hash_embed(text: str, dim: int) -> np.ndarray:
    """Unigrams + bigrams hashed into ``dim`` signed buckets, 1+log(tf) weights, L2-normalised."""
    words = tokenize(text)
    feats: Counter[str] = Counter(words)
    feats.update(f"{a}_{b}" for a, b in zip(words, words[1:], strict=False))
    vec = np.zeros(dim, dtype=np.float32)
    for feat, tf in feats.items():
        h = zlib.crc32(feat.encode("utf-8"))
        vec[h % dim] += (1.0 if (h >> 31) & 1 else -1.0) * (1.0 + math.log(tf))
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def hash_embed_many(texts: list[str], dim: int) -> np.ndarray:
    out = np.empty((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i] = hash_embed(t, dim)
    return out


def _worker(args: tuple[list[str], int]) -> np.ndarray:
    texts, dim = args
    return hash_embed_many(texts, dim)


def hash_embed_parallel(texts: list[str], dim: int, workers: int, batch_size: int) -> np.ndarray:
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_worker, [(b, dim) for b in batches]))
    return np.vstack(parts) if parts else np.empty((0, dim), dtype=np.float32)

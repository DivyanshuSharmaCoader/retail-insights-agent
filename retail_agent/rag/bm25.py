"""Okapi BM25 keyword scoring over the indexed chunks (the sparse half of hybrid search).

Dense vectors are good at paraphrases; BM25 is good at exact terms such as SKU codes,
campaign names ("Summer Chill") or numbers. Hybrid retrieval uses both.
"""

from __future__ import annotations

import math
from collections import Counter

from retail_agent.text import tokenize


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_tfs = [Counter(tokenize(t)) for t in texts]
        self.doc_lens = [sum(tf.values()) for tf in self.doc_tfs]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0
        df: Counter[str] = Counter()
        for tf in self.doc_tfs:
            df.update(tf.keys())
        n = len(texts)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = [t for t in tokenize(query) if t in self.idf]
        out = []
        for tf, dl in zip(self.doc_tfs, self.doc_lens, strict=True):
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                    )
            out.append(s)
        return out

    def top(self, query: str, n: int) -> list[tuple[int, float]]:
        scored = [(i, s) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

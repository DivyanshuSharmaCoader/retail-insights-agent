"""Offline retrieval evaluation on a labelled question set (no LLM or API key needed).

Compares dense-only MMR with hybrid (dense + BM25, RRF fusion, relevance gate, MMR) and
reports source-level hit@1 / hit@k, MRR, section-level hit@k and off-topic rejection.

    python -m scripts.eval_retrieval
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from retail_agent.config import PROJECT_ROOT, get_settings
from retail_agent.rag.ingest import build_index
from retail_agent.rag.retriever import Retriever

GOLDEN = PROJECT_ROOT / "data" / "eval" / "retrieval_golden.jsonl"


def load_golden(path: Path = GOLDEN) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def evaluate(retriever: Retriever, golden: list[dict], k: int) -> dict:
    hit1 = hitk = sect = rr = 0.0
    in_domain = [g for g in golden if g["sources"]]
    off_topic = [g for g in golden if not g["sources"]]
    misses = []
    for g in in_domain:
        hits = await retriever.aretrieve(g["question"], k=k)
        sources = [h.chunk.source for h in hits]
        ranks = [i for i, s in enumerate(sources) if s in g["sources"]]
        if ranks:
            hitk += 1
            rr += 1 / (ranks[0] + 1)
            hit1 += ranks[0] == 0
        else:
            misses.append(g["question"])
        section_ok = not g["section"] or any(
            h.chunk.source in g["sources"] and g["section"] in h.chunk.section for h in hits
        )
        sect += 1 if ranks and section_ok else 0
    rejected = 0
    for g in off_topic:
        rejected += not await retriever.aretrieve(g["question"], k=k)
    n = len(in_domain)
    return {
        "questions": n,
        "hit@1": round(hit1 / n, 3),
        f"hit@{k}": round(hitk / n, 3),
        "mrr": round(rr / n, 3),
        f"section_hit@{k}": round(sect / n, 3),
        "off_topic_rejected": f"{rejected}/{len(off_topic)}",
        "misses": misses,
    }


async def main() -> None:
    settings = get_settings()
    store, _ = await build_index(settings, save=True)
    golden = load_golden()
    k = settings.retrieval_k
    modes = (("dense-only (hashing + MMR)", False), ("hybrid (dense + BM25, RRF)", True))
    for name, hybrid in modes:
        r = Retriever.from_settings(settings, store=store)
        r.hybrid = hybrid
        print(name, json.dumps(await evaluate(r, golden, k), indent=1))


if __name__ == "__main__":
    asyncio.run(main())

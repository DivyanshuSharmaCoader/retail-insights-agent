"""Profile and benchmark local embedding (no API key needed).

1. cProfile the sequential hashing-embedding path to see where the time goes.
2. Time sequential vs a 4-process pool at several corpus sizes to find where
   multiprocessing starts to pay off (this set ``parallel_embed_min_chunks``).

    python -m scripts.profile_ingest
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import random
import time

from retail_agent.rag.hashing import hash_embed_many, hash_embed_parallel

# Heavy imports (pydantic-settings, LangChain splitters) happen inside main(): with the
# "spawn" start method (macOS/Windows default) every worker re-imports this module, and
# top-level LangChain imports added ~1 s to each worker's start-up in our measurements.

VOCAB = ("store dairy snacks beverages revenue promotion supplier fill rate return policy margin "
         "markdown region south north west east chiller temperature penalty uplift baseline "
         "campaign category product units discount weekend shrinkage").split()


def synthetic_chunks(n: int, words_per_chunk: int = 110) -> list[str]:
    rng = random.Random(0)
    return [" ".join(rng.choices(VOCAB, k=words_per_chunk)) + f" doc{i}" for i in range(n)]


def main() -> None:
    from retail_agent.config import get_settings
    from retail_agent.rag.ingest import chunk_documents, load_documents, strip_boilerplate

    settings = get_settings()
    dim = settings.local_embedding_dim
    real = [c.text for c in chunk_documents(strip_boilerplate(load_documents(settings.docs_dir)),
                                            settings.chunk_size, settings.chunk_overlap)[0]]

    profiler = cProfile.Profile()
    profiler.enable()
    hash_embed_many(synthetic_chunks(3000), dim)
    profiler.disable()
    out = io.StringIO()
    pstats.Stats(profiler, stream=out).sort_stats("tottime").print_stats(5)
    lines = [line for line in out.getvalue().splitlines() if line.strip()]
    print("cProfile of 3,000 chunks, top functions by own time:")
    print("\n".join(lines[-6:]))

    t = time.perf_counter()
    hash_embed_many(real, dim)
    elapsed_ms = (time.perf_counter() - t) * 1000
    print(f"\nreal corpus: {len(real)} chunks embedded in {elapsed_ms:.1f} ms")

    workers = min(4, os.cpu_count() or 1)
    print(f"\nsequential vs {workers} worker processes:")
    for n in (2_000, 10_000, 30_000, 100_000):
        texts = synthetic_chunks(n)
        t = time.perf_counter()
        hash_embed_many(texts, dim)
        seq = time.perf_counter() - t
        t = time.perf_counter()
        hash_embed_parallel(texts, dim, workers, batch_size=1024)
        par = time.perf_counter() - t
        print(f"  {n:>7,} chunks: sequential {seq:5.2f}s | pool {par:5.2f}s | {seq / par:4.2f}x")


if __name__ == "__main__":
    main()

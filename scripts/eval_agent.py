"""End-to-end agent evaluation against a real LLM (needs an API key in .env).

For each labelled question it checks:
* route accuracy (the router picked an allowed route),
* SQL success (a query ran, for sql/both routes),
* answer correctness: either the reference SQL value appears in the agent's result rows
  (within 1%, also accepting a fraction vs percentage), or the expected text is in the answer,
* verification (every citation points at real evidence),
and reports latency (mean / p95). Questions run concurrently, capped by a semaphore.

    python -m scripts.eval_agent
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time

from sqlalchemy import text

from retail_agent.agent.graph import ask, build_graph
from retail_agent.bootstrap import build_deps, ensure_data
from retail_agent.config import PROJECT_ROOT, get_settings

GOLDEN = PROJECT_ROOT / "data" / "eval" / "agent_golden.jsonl"


def _numbers(rows: list[list]) -> list[float]:
    out = []
    for row in rows:
        for v in row:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
    return out


def _matches(ref, rows: list[list], answer: str) -> bool:
    if isinstance(ref, str):
        return any(ref in str(v) for row in rows for v in row) or ref in answer
    for v in _numbers(rows):
        for candidate in (v, v * 100, v / 100):
            if ref == 0 and candidate == 0 or ref and abs(candidate - ref) / abs(ref) <= 0.01:
                return True
    return False


async def main() -> None:
    settings = get_settings()
    await ensure_data(settings)
    deps = await build_deps(settings)
    if deps.llm is None:
        raise SystemExit("No LLM configured: copy .env.example to .env and set OPENAI_API_KEY "
                         "(or LLM_PROVIDER=gemini with GOOGLE_API_KEY).")
    graph = build_graph(deps)
    golden = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    sem = asyncio.Semaphore(3)  # stay well inside provider rate limits

    async def run(case: dict) -> dict:
        async with sem:
            start = time.perf_counter()
            try:
                state = await ask(graph, case["question"], timeout_s=settings.request_timeout_s)
            except Exception as exc:
                return {"question": case["question"], "error": type(exc).__name__}
            latency = time.perf_counter() - start
        rows = (state.get("sql_result") or {}).get("rows", [])
        answer = state.get("answer", "")
        if "reference_sql" in case:
            with deps.sql_runner.engine.connect() as conn:
                ref = conn.execute(text(case["reference_sql"])).scalar()
            ref = float(ref) if not isinstance(ref, str) else ref
            correct = _matches(ref, rows, answer)
        else:
            correct = all(t.lower() in answer.lower() for t in case.get("expect_text", []))
        return {
            "question": case["question"],
            "route_ok": state.get("route") in case["routes"],
            "sql_ok": (state.get("sql_result") is not None
                       if state.get("route") in ("sql", "both") else None),
            "correct": correct,
            "verified": bool(state.get("verified")),
            "sql_attempts": state.get("sql_attempts", 0),
            "latency_s": round(latency, 2),
        }

    results = await asyncio.gather(*(run(c) for c in golden))
    ok = [r for r in results if "error" not in r]
    sql_cases = [r for r in ok if r["sql_ok"] is not None]
    latencies = sorted(r["latency_s"] for r in ok)

    def rate(xs: list[bool]) -> str:
        return f"{sum(xs)}/{len(xs)}"

    for r in results:
        print(json.dumps(r))
    print("\nsummary:", json.dumps({
        "questions": len(results),
        "errors": len(results) - len(ok),
        "route_accuracy": rate([r["route_ok"] for r in ok]),
        "sql_success": rate([r["sql_ok"] for r in sql_cases]),
        "answer_correct": rate([r["correct"] for r in ok]),
        "verified": rate([r["verified"] for r in ok]),
        "sql_self_corrections": sum(max(0, r["sql_attempts"] - 1) for r in ok),
        "latency_mean_s": round(statistics.mean(latencies), 2) if latencies else None,
        "latency_p95_s": latencies[int(0.95 * (len(latencies) - 1))] if latencies else None,
    }, indent=1))


if __name__ == "__main__":
    asyncio.run(main())

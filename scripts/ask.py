"""Ask the agent from the terminal (needs OPENAI_API_KEY or GOOGLE_API_KEY in .env).

    python -m scripts.ask "Did the Summer Chill promotion meet its uplift target?"
    python -m scripts.ask            # interactive chat; follow-up questions keep context
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from langgraph.checkpoint.memory import InMemorySaver

from retail_agent.agent.graph import ask, build_graph
from retail_agent.bootstrap import build_deps, ensure_data
from retail_agent.config import get_settings


def show(state: dict) -> None:
    print(f"\nroute: {state.get('route')}  |  question: {state.get('standalone_question')}")
    result = state.get("sql_result")
    if result:
        print(f"\nSQL ({state.get('sql_attempts')} attempt(s)):\n{result['sql']}")
        print(" | ".join(result["columns"]))
        for row in result["rows"][:10]:
            print(" | ".join(str(v) for v in row))
        if result["row_count"] > 10:
            print(f"... {result['row_count']} rows")
    for c in state.get("contexts") or []:
        print(f"[{c['id']}] {c['source']} > {c['section']}  (score {c['score']})")
    print(f"\n{state.get('answer')}")
    if state.get("warnings"):
        print("\nwarnings:", "; ".join(state["warnings"]))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="omit for an interactive session")
    args = parser.parse_args()

    settings = get_settings()
    await ensure_data(settings)
    deps = await build_deps(settings)
    if deps.llm is None:
        raise SystemExit("No LLM configured: copy .env.example to .env and set OPENAI_API_KEY "
                         "(or LLM_PROVIDER=gemini with GOOGLE_API_KEY).")
    if args.question:
        show(await ask(build_graph(deps), args.question, timeout_s=settings.request_timeout_s))
        return

    graph = build_graph(deps, checkpointer=InMemorySaver())
    thread = uuid.uuid4().hex[:8]
    print("Ask about Kestrel Retail's sales or policies. Empty line to quit.")
    while question := (await asyncio.to_thread(input, "\n> ")).strip():
        show(await ask(graph, question, thread_id=thread, timeout_s=settings.request_timeout_s))


if __name__ == "__main__":
    asyncio.run(main())

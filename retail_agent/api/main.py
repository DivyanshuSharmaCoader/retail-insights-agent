"""Async FastAPI service around the agent.

Endpoints
---------
GET  /health              liveness plus index size and cache backend
GET  /v1/schema           the schema description the SQL generator sees
POST /v1/ask              one question -> answer, route, SQL, rows, sources, warnings
POST /v1/ask/stream       the same, streamed as Server-Sent Events (one event per graph step)
POST /v1/index/rebuild    re-ingest the documents in the background, then hot-swap the index

Run:  uvicorn retail_agent.api.main:app --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from retail_agent.agent.graph import AgentDeps, ask, build_graph
from retail_agent.bootstrap import build_deps, cache_signature, ensure_data
from retail_agent.cache import AnswerCache
from retail_agent.config import Settings, get_settings
from retail_agent.db.sql_tools import describe_schema
from retail_agent.logging_setup import configure_logging, log_event, request_id_var
from retail_agent.rag.ingest import build_index
from retail_agent.rag.retriever import Retriever

log = logging.getLogger("retail_agent.api")


# ----------------------------------------------------------------------------- schemas
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    thread_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$",
        description="Pass the same id to ask follow-up questions in one conversation.",
    )


class Source(BaseModel):
    id: str
    source: str
    section: str
    score: float


class AskResponse(BaseModel):
    answer: str
    route: str
    standalone_question: str
    sql: str | None = None
    sql_columns: list[str] = []
    sql_rows: list[list[Any]] = []
    sources: list[Source] = []
    verified: bool
    warnings: list[str] = []
    cached: bool = False
    thread_id: str | None = None
    latency_ms: float = 0.0


# ----------------------------------------------------------------------------- wiring
@dataclass
class Services:
    settings: Settings
    deps: AgentDeps
    graph: Any | None  # stateless (no checkpointer): safe to cache; None if no LLM key
    chat_graph: Any | None  # with a checkpointer: remembers each thread_id's conversation
    cache: AnswerCache
    signature: str
    index_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    background: set[asyncio.Task] = field(default_factory=set)


def _to_response(state: dict, **extra: Any) -> AskResponse:
    result = state.get("sql_result") or {}
    return AskResponse(
        answer=state.get("answer", ""),
        route=state.get("route", ""),
        standalone_question=state.get("standalone_question") or state.get("question", ""),
        sql=result.get("sql") or state.get("sql"),
        sql_columns=result.get("columns", []),
        sql_rows=result.get("rows", [])[:50],
        sources=[Source(id=c["id"], source=c["source"], section=c["section"], score=c["score"])
                 for c in state.get("contexts") or []],
        verified=bool(state.get("verified")),
        warnings=state.get("warnings") or [],
        **extra,
    )


def create_app(settings: Settings | None = None, llm: BaseChatModel | None = None) -> FastAPI:
    """App factory; tests pass their own settings and a scripted chat model."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = settings or get_settings()
        configure_logging(cfg.log_level)
        if cfg.auto_setup:
            await ensure_data(cfg)
        deps = await build_deps(cfg, llm)
        engine = deps.sql_runner.engine
        app.state.services = Services(
            settings=cfg,
            deps=deps,
            graph=build_graph(deps) if deps.llm else None,
            chat_graph=build_graph(deps, checkpointer=InMemorySaver()) if deps.llm else None,
            cache=AnswerCache.from_url(cfg.redis_url, ttl_s=cfg.cache_ttl_s,
                                       maxsize=cfg.memory_cache_size),
            signature=cache_signature(cfg),
        )
        log_event(log, "service ready", chunks=len(deps.retriever.store), db=engine.dialect.name,
                  cache=app.state.services.cache.backend)
        yield
        await app.state.services.cache.close()
        engine.dispose()

    app = FastAPI(title="Retail Insights Agent", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["x-request-id"] = rid
        log_event(log, "request", method=request.method, path=request.url.path,
                  status=response.status_code, request_id=rid,
                  latency_ms=round((time.perf_counter() - start) * 1000, 1))
        return response

    def services(request: Request) -> Services:
        return request.app.state.services

    def require_llm(s: Services) -> None:
        if s.graph is None:
            raise HTTPException(503, "LLM not configured: set OPENAI_API_KEY (or GOOGLE_API_KEY "
                                     "with LLM_PROVIDER=gemini) and restart.")

    @app.get("/health")
    async def health(request: Request) -> dict:
        s = services(request)
        return {"status": "ok", "index_chunks": len(s.deps.retriever.store),
                "documents": s.deps.documents, "cache": s.cache.backend,
                "cache_stats": s.cache.stats, "llm_provider": s.settings.llm_provider,
                "llm_ready": s.graph is not None}

    @app.get("/v1/schema")
    async def schema(request: Request) -> dict:
        s = services(request)
        return {"schema": await asyncio.to_thread(describe_schema, s.deps.sql_runner.engine)}

    @app.post("/v1/ask", response_model=AskResponse)
    async def ask_endpoint(body: AskRequest, request: Request) -> AskResponse:
        s = services(request)
        require_llm(s)
        start = time.perf_counter()
        key = s.cache.key_for(body.question, s.signature)
        if body.thread_id is None and (hit := await s.cache.get(key)) is not None:
            latency = round((time.perf_counter() - start) * 1000, 1)
            return AskResponse(**{**hit, "cached": True, "latency_ms": latency})

        graph = s.chat_graph if body.thread_id else s.graph
        try:
            state = await ask(graph, body.question, body.thread_id, s.settings.request_timeout_s)
        except TimeoutError as exc:
            raise HTTPException(504, "The agent took too long to answer.") from exc
        except Exception as exc:  # provider/network errors surface as 502, with a log entry
            log.exception("agent failed")
            raise HTTPException(502, f"Upstream model error: {type(exc).__name__}") from exc

        response = _to_response(state, thread_id=body.thread_id,
                                latency_ms=round((time.perf_counter() - start) * 1000, 1))
        if body.thread_id is None and response.verified and response.route != "out_of_scope":
            await s.cache.set(key, response.model_dump(exclude={"cached", "latency_ms",
                                                                "thread_id"}))
        log_event(log, "answered", route=response.route, verified=response.verified,
                  sql_attempts=state.get("sql_attempts", 0), sources=len(response.sources),
                  latency_ms=response.latency_ms)
        return response

    @app.post("/v1/ask/stream")
    async def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
        s = services(request)
        require_llm(s)
        graph = s.chat_graph if body.thread_id else s.graph
        config = {"configurable": {"thread_id": body.thread_id}} if body.thread_id else {}
        inputs = {"messages": [HumanMessage(content=body.question)]}

        async def events():
            start = time.perf_counter()
            final: dict[str, Any] = {}
            try:
                async for namespace, update in graph.astream(
                    inputs, config=config, stream_mode="updates", subgraphs=True
                ):
                    for node, delta in update.items():
                        delta = delta or {}
                        if not namespace:
                            final.update({k: v for k, v in delta.items() if k != "messages"})
                        step = {"node": node, **{k: delta[k] for k in
                                ("route", "standalone_question", "sql", "sql_error", "answer")
                                if delta.get(k)}}
                        if "contexts" in delta:
                            step["sources"] = [c["source"] for c in delta["contexts"]]
                        yield f"event: step\ndata: {json.dumps(step, default=str)}\n\n"
            except Exception as exc:
                log.exception("stream failed")
                yield f"event: error\ndata: {json.dumps({'error': type(exc).__name__})}\n\n"
                return
            resp = _to_response(final, thread_id=body.thread_id,
                                latency_ms=round((time.perf_counter() - start) * 1000, 1))
            yield f"event: answer\ndata: {resp.model_dump_json()}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    @app.post("/v1/index/rebuild", status_code=202)
    async def rebuild(request: Request) -> dict:
        s = services(request)
        if s.index_lock.locked():
            return {"status": "already rebuilding"}

        async def job() -> None:
            async with s.index_lock:
                store, stats = await build_index(s.settings)
                s.deps.retriever = Retriever.from_settings(s.settings, store=store)
                s.deps.documents = sorted({c.source for c in store.chunks})
                s.signature = cache_signature(s.settings)  # old cache entries no longer match
                log_event(log, "index swapped", chunks=stats.chunks, total_ms=stats.total_ms)

        task = asyncio.create_task(job())
        s.background.add(task)  # keep a reference so the task is not garbage-collected
        task.add_done_callback(s.background.discard)
        return {"status": "rebuilding"}

    return app


app = create_app()

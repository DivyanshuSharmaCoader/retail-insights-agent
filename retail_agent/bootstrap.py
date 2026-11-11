"""Shared start-up used by the API and the CLI scripts."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

from retail_agent.agent.graph import AgentDeps
from retail_agent.config import Settings
from retail_agent.db.seed import is_seeded, seed_database
from retail_agent.db.sql_tools import SqlRunner, readonly_engine
from retail_agent.llm import embedding_signature, get_chat_model, model_signature
from retail_agent.rag.ingest import build_index
from retail_agent.rag.retriever import IndexMismatchError, Retriever

log = logging.getLogger(__name__)


async def ensure_data(settings: Settings) -> None:
    """Seed the database and build the index if they are missing or stale."""
    url = make_url(settings.database_url)
    if url.get_backend_name() == "sqlite" and url.database:
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(settings.database_url)
    try:
        if not await asyncio.to_thread(is_seeded, engine):
            await asyncio.to_thread(seed_database, engine)
    finally:
        engine.dispose()
    manifest = settings.index_dir / "manifest.json"
    stale = (not manifest.exists()
             or json.loads(manifest.read_text()).get("embedding") != embedding_signature(settings))
    if stale:
        await build_index(settings)


async def build_deps(settings: Settings, llm: BaseChatModel | None = None) -> AgentDeps:
    try:
        retriever = Retriever.from_settings(settings)
    except IndexMismatchError:
        store, _ = await build_index(settings)
        retriever = Retriever.from_settings(settings, store=store)
    if llm is None:
        try:
            llm = get_chat_model(settings)
        except Exception as exc:  # e.g. missing API key: start anyway, /v1/ask returns 503
            log.warning("LLM not configured", extra={"fields": {"error": str(exc)[:200]}})
    return AgentDeps(
        llm=llm,
        retriever=retriever,
        sql_runner=SqlRunner(readonly_engine(settings.database_url), settings.sql_row_limit,
                             settings.sql_timeout_s),
        documents=sorted({c.source for c in retriever.store.chunks}),
        max_sql_attempts=settings.max_sql_attempts,
    )


def cache_signature(settings: Settings) -> str:
    """Changes whenever the model, the embedding space or the index changes."""
    manifest = settings.index_dir / "manifest.json"
    built = json.loads(manifest.read_text()).get("built_at", "") if manifest.exists() else ""
    return f"{model_signature(settings)}|{embedding_signature(settings)}|{built}"

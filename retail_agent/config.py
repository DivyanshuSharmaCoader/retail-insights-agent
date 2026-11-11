"""Application settings, read from environment variables or a local .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM -----------------------------------------------------------------
    llm_provider: Literal["openai", "gemini"] = "openai"
    openai_model: str = "gpt-4.1-mini"
    gemini_model: str = "gemini-2.5-flash"
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2

    # --- Embeddings ------------------------------------------------------------
    # "local" = deterministic hashing embeddings: free, offline, no API key.
    embedding_provider: Literal["local", "openai", "gemini"] = "local"
    openai_embedding_model: str = "text-embedding-3-small"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    local_embedding_dim: int = 1024

    # --- Data ------------------------------------------------------------------
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'data' / 'retail.db'}"
    docs_dir: Path = PROJECT_ROOT / "data" / "docs"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    auto_setup: bool = True  # seed the database and build the index on startup if missing

    # --- RAG -------------------------------------------------------------------
    chunk_size: int = 700
    chunk_overlap: int = 100
    retrieval_k: int = 4
    retrieval_fetch_k: int = 12
    mmr_lambda: float = 0.6  # 1.0 = pure relevance, 0.0 = pure diversity
    # Relevance gate (hybrid): a chunk is kept when cos/gate_dense + bm25/gate_bm25 >= 1,
    # i.e. either signal alone at its floor, or a mix of both. Calibrated on
    # data/eval/retrieval_golden.jsonl so off-topic questions retrieve nothing.
    gate_dense: float = 0.20
    gate_bm25: float = 6.0
    dense_floor: float = 0.12  # cosine floor used when hybrid search is switched off
    hybrid_search: bool = True
    embed_workers: int = 4
    parallel_embed_min_chunks: int = 5000  # measured break-even for 4 worker processes

    # --- SQL -------------------------------------------------------------------
    sql_row_limit: int = 200
    sql_timeout_s: float = 10.0
    max_sql_attempts: int = 3

    # --- Cache -----------------------------------------------------------------
    redis_url: str | None = None  # e.g. redis://localhost:6379/0; None = in-memory only
    cache_ttl_s: int = 3600
    memory_cache_size: int = 256

    # --- API -------------------------------------------------------------------
    request_timeout_s: float = 90.0
    log_level: str = "INFO"
    max_question_chars: int = Field(default=500, ge=10)


@lru_cache
def get_settings() -> Settings:
    return Settings()

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import create_engine

from retail_agent.config import Settings
from retail_agent.db.seed import seed_database
from retail_agent.db.sql_tools import SqlRunner, readonly_engine
from retail_agent.rag.ingest import build_index
from retail_agent.rag.retriever import Retriever

DAIRY_MAY_RETURN_RATE_SQL = """
SELECT ROUND(100.0 * SUM(COALESCE(r.units, 0)) / SUM(s.units), 2) AS return_rate_pct
FROM sales s
JOIN products p ON p.product_id = s.product_id
LEFT JOIN (SELECT sale_id, SUM(units) AS units FROM returns GROUP BY sale_id) r
  ON r.sale_id = s.sale_id
WHERE p.category = 'Dairy' AND s.sale_date BETWEEN '2026-05-01' AND '2026-05-31'
"""


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    root = tmp_path_factory.mktemp("agent")
    cfg = Settings(
        database_url=f"sqlite:///{root / 'retail.db'}",
        index_dir=root / "index",
        redis_url=None,
        embed_workers=1,
        auto_setup=True,
        llm_provider="openai",
        embedding_provider="local",
    )
    seed_database(create_engine(cfg.database_url))
    asyncio.run(build_index(cfg))
    return cfg


@pytest.fixture
def runner(settings: Settings) -> SqlRunner:
    return SqlRunner(readonly_engine(settings.database_url), row_limit=200, timeout_s=5)


@pytest.fixture
def retriever(settings: Settings) -> Retriever:
    return Retriever.from_settings(settings)

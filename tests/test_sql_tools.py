from __future__ import annotations

import time

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from retail_agent.db.sql_tools import SQLValidationError, validate_sql
from tests.conftest import DAIRY_MAY_RETURN_RATE_SQL


@pytest.mark.parametrize(
    "query, message",
    [
        ("DELETE FROM sales", "read-only"),
        ("UPDATE products SET unit_price = 0", "read-only"),
        ("DROP TABLE sales", "read-only"),
        ("SELECT 1; DROP TABLE sales", "one SQL statement"),
        ("SELECT * FROM customers", "Unknown table"),
        ("SELECT * FROM sqlite_master", "Unknown table"),
        ("", "Empty"),
    ],
)
def test_rejects_unsafe_or_invalid_sql(query: str, message: str) -> None:
    with pytest.raises(SQLValidationError, match=message):
        validate_sql(query, "sqlite", row_limit=200)


def test_adds_and_caps_limit() -> None:
    assert validate_sql("SELECT * FROM sales", "sqlite", 200).endswith("LIMIT 200")
    assert validate_sql("SELECT * FROM sales LIMIT 10", "sqlite", 200).endswith("LIMIT 10")
    assert validate_sql("SELECT * FROM sales LIMIT 9999", "sqlite", 200).endswith("LIMIT 200")


def test_allows_ctes_and_caps_unions() -> None:
    cte = "WITH m AS (SELECT store_id, SUM(revenue) r FROM sales GROUP BY 1) SELECT * FROM m"
    assert "LIMIT 200" in validate_sql(cte, "sqlite", 200)
    union = "SELECT store_id FROM stores UNION SELECT store_id FROM sales"
    assert validate_sql(union, "sqlite", 50).endswith("LIMIT 50")


async def test_runs_query_and_returns_json_safe_rows(runner) -> None:
    result = await runner.run(DAIRY_MAY_RETURN_RATE_SQL)
    assert result.columns == ["return_rate_pct"]
    rate = result.rows[0][0]
    assert 5.0 < rate < 8.0  # the planted May cold-chain spike
    assert "return_rate_pct" in result.to_markdown()


async def test_readonly_connection_refuses_writes(runner) -> None:
    with pytest.raises(OperationalError), runner.engine.connect() as conn:
        conn.execute(text("DELETE FROM sales"))


async def test_query_timeout(runner, monkeypatch) -> None:
    runner.timeout_s = 0.1
    monkeypatch.setattr(runner, "_execute_sync", lambda sql: time.sleep(0.5))
    with pytest.raises(TimeoutError):
        await runner.run("SELECT 1 FROM stores")

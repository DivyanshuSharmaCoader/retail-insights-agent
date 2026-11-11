"""Safe text-to-SQL execution: schema description, SQL validation and read-only execution.

Defence in depth for LLM-written SQL:
1. ``validate_sql`` parses the query with sqlglot and accepts exactly one read-only SELECT
   over known tables, then enforces a row LIMIT.
2. ``SqlRunner`` executes on a read-only connection (SQLite ``mode=ro`` / Postgres
   ``READ ONLY`` transaction with a statement timeout), in a worker thread so the event
   loop is never blocked.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import decimal
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import sqlglot
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlglot import exp
from sqlglot.errors import ParseError

from retail_agent.db.models import ALLOWED_TABLES, metadata

_WRITE_NODES = tuple(
    getattr(exp, name)
    for name in ("Insert", "Update", "Delete", "Drop", "Create", "Alter", "Merge",
                 "TruncateTable", "Command", "Pragma", "Grant", "Copy")
    if hasattr(exp, name)
)
_SET_OP = getattr(exp, "SetOperation", exp.Union)


class SQLValidationError(ValueError):
    """Raised when generated SQL is unsafe or invalid; the message is shown to the LLM."""


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    elapsed_ms: float
    row_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.row_count = len(self.rows)

    def to_markdown(self, max_rows: int = 30) -> str:
        if not self.rows:
            return "(query returned no rows)"
        head = "| " + " | ".join(self.columns) + " |"
        sep = "|" + "---|" * len(self.columns)
        body = ["| " + " | ".join(_fmt(v) for v in r) + " |" for r in self.rows[:max_rows]]
        more = []
        if self.row_count > max_rows or self.truncated:
            more.append(f"(showing {min(max_rows, self.row_count)} of {self.row_count}"
                        f"{'+' if self.truncated else ''} rows)")
        return "\n".join([head, sep, *body, *more])


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) >= 100 else f"{v:.4g}"
    return str(v)


def _json_safe(v: Any) -> Any:
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, dt.date | dt.datetime):
        return v.isoformat()
    return v


def validate_sql(sql: str, dialect: str, row_limit: int) -> str:
    """Return a safe, LIMIT-ed version of ``sql`` or raise ``SQLValidationError``."""
    cleaned = sql.strip().strip(";").strip()
    if not cleaned:
        raise SQLValidationError("Empty query.")
    try:
        statements = [s for s in sqlglot.parse(cleaned, read=dialect) if s is not None]
    except ParseError as exc:
        raise SQLValidationError(f"SQL syntax error: {str(exc).splitlines()[0]}") from exc
    if len(statements) != 1:
        raise SQLValidationError("Exactly one SQL statement is allowed.")
    tree = statements[0]

    if not isinstance(tree, exp.Select | _SET_OP):
        raise SQLValidationError("Only read-only SELECT queries are allowed.")
    if any(isinstance(node, _WRITE_NODES) for node in tree.walk()):
        raise SQLValidationError("Only read-only SELECT queries are allowed.")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    unknown = sorted(
        {t.name.lower() for t in tree.find_all(exp.Table)} - ALLOWED_TABLES - cte_names
    )
    if unknown:
        raise SQLValidationError(
            f"Unknown table(s): {', '.join(unknown)}. "
            f"Available tables: {', '.join(sorted(ALLOWED_TABLES))}."
        )

    limit = tree.args.get("limit")
    if isinstance(tree, exp.Select):
        current = None
        if limit is not None and isinstance(limit.expression, exp.Literal):
            current = int(limit.expression.this)
        if limit is None or current is None or current > row_limit:
            tree = tree.limit(row_limit if current is None else min(current, row_limit))
    else:  # UNION / INTERSECT / EXCEPT: cap with an outer query
        tree = exp.select("*").from_(tree.subquery("q")).limit(row_limit)
    return tree.sql(dialect=dialect)


@lru_cache(maxsize=8)
def _sample_rows(url: str, table: str, n: int = 3) -> tuple[tuple[str, ...], ...]:
    eng = create_engine(url)
    try:
        with eng.connect() as conn:
            res = conn.execute(text(f"SELECT * FROM {table} LIMIT {n}"))  # noqa: S608 - fixed names
            return tuple(tuple(str(_json_safe(v)) for v in row) for row in res)
    finally:
        eng.dispose()


def describe_schema(engine: Engine, sample_rows: int = 3) -> str:
    """Compact schema text for the LLM: columns, types, comments, keys and sample rows.

    Sample rows are cached per database URL (``lru_cache``) because the schema prompt is
    rebuilt on every SQL generation.
    """
    url = engine.url.render_as_string(hide_password=False)
    parts = []
    for table in metadata.sorted_tables:
        cols = []
        for c in table.columns:
            bits = [f"{c.name} {c.type}"]
            if c.primary_key:
                bits.append("PK")
            for fk in c.foreign_keys:
                bits.append(f"FK->{fk.target_fullname}")
            if c.comment:
                bits.append(f"-- {c.comment}")
            cols.append("  " + " ".join(bits))
        block = [f"TABLE {table.name}: {table.comment or ''}", *cols]
        if sample_rows:
            samples = _sample_rows(url, table.name, sample_rows)
            if samples:
                block.append("  sample rows: " + "; ".join("(" + ", ".join(r) + ")"
                                                          for r in samples))
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def readonly_engine(database_url: str) -> Engine:
    """Engine whose connections cannot write (defence in depth behind ``validate_sql``)."""
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite" and url.database:
        ro = f"sqlite:///file:{url.database}?mode=ro&uri=true"
        return create_engine(ro, connect_args={"check_same_thread": False})
    return create_engine(database_url, pool_pre_ping=True, pool_size=5, max_overflow=5)


class SqlRunner:
    def __init__(self, engine: Engine, row_limit: int = 200, timeout_s: float = 10.0) -> None:
        self.engine = engine
        self.row_limit = row_limit
        self.timeout_s = timeout_s
        self.dialect = "postgres" if engine.dialect.name == "postgresql" else engine.dialect.name

    def validate(self, sql: str) -> str:
        return validate_sql(sql, self.dialect, self.row_limit)

    def _execute_sync(self, safe_sql: str) -> QueryResult:
        start = time.perf_counter()
        with self.engine.connect() as conn:
            if self.engine.dialect.name == "postgresql":
                conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                conn.exec_driver_sql(f"SET LOCAL statement_timeout = {int(self.timeout_s * 1000)}")
            res = conn.execute(text(safe_sql))
            columns = list(res.keys())
            rows = [[_json_safe(v) for v in row] for row in res.fetchmany(self.row_limit + 1)]
            conn.rollback()
        truncated = len(rows) > self.row_limit
        return QueryResult(
            sql=safe_sql,
            columns=columns,
            rows=rows[: self.row_limit],
            truncated=truncated,
            elapsed_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    async def run(self, sql: str) -> QueryResult:
        """Validate, then execute in a worker thread with an overall timeout."""
        safe_sql = self.validate(sql)
        return await asyncio.wait_for(
            asyncio.to_thread(self._execute_sync, safe_sql), timeout=self.timeout_s
        )

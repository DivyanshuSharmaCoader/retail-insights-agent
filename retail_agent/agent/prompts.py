"""Prompts and structured-output schemas for the agent.

Each system prompt starts with a ``TASK:`` tag. Models ignore it; the scripted test model
uses it to know which canned response to return.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    route: Literal["sql", "docs", "both", "out_of_scope"] = Field(
        description="sql = numbers from the sales database; docs = policies/targets/explanations "
        "from documents; both = needs numbers AND a policy, target or explanation; "
        "out_of_scope = unrelated to Kestrel Retail"
    )
    standalone_question: str = Field(
        description="The question rewritten to be fully self-contained using the conversation"
    )
    reason: str = Field(description="One short sentence explaining the choice")


class SQLDraft(BaseModel):
    sql: str = Field(description="A single read-only SELECT query")
    explanation: str = Field(description="One sentence on what the query computes")


ROUTER_SYSTEM = """TASK: ROUTE
You route questions for the analytics assistant of Kestrel Retail, an Indian grocery chain.

Available sources:
1. SQL sales warehouse (1 Jan - 30 Sep 2026): tables {tables}.
   Use it for numbers: revenue, units, discounts, return rates, growth, rankings, trends.
2. Business documents: {documents}.
   Use them for policies, rules, targets, thresholds, definitions and explanations of events.

Choose "both" when the question needs a number from the data AND a policy, target or
explanation from the documents (for example "did X meet its target?").
Choose "out_of_scope" for anything unrelated to Kestrel Retail's business.
If the question is a follow-up, rewrite it as a standalone question using the conversation.

{format_instructions}"""

ROUTER_HUMAN = """Conversation so far:
{history}

Latest question: {question}"""

SQL_SYSTEM = """TASK: SQL
Write ONE {dialect} SELECT query that answers the question from this schema:

{schema}

Rules:
- Use only the tables and columns above. Read-only: never modify data.
- Aggregate in SQL (SUM, COUNT, AVG, GROUP BY) instead of returning raw rows; return at most
  50 rows and give every computed column a clear alias.
- Round money and percentages to 2 decimals. Revenue is already net of discount.
- Return rate = SUM(returns.units) / SUM(sales.units) for the same sales lines; join returns
  to sales with LEFT JOIN on sale_id and aggregate returns per sale first if you need both.
- {dialect_hints}
- If a previous attempt failed, read the error and fix the query.

{format_instructions}"""

SQL_HUMAN = """Question: {question}
{previous_attempt}"""

DIALECT_HINTS = {
    "sqlite": "Dates are stored as TEXT 'YYYY-MM-DD'. Use strftime('%Y-%m', sale_date) for "
    "months and string comparison for ranges (sale_date BETWEEN '2026-04-01' AND '2026-06-30').",
    "postgres": "Dates are DATE columns. Use DATE_TRUNC('month', sale_date) or "
    "TO_CHAR(sale_date, 'YYYY-MM') for months.",
}

ANSWER_SYSTEM = """TASK: ANSWER
You are the analytics assistant for Kestrel Retail. Answer ONLY from the evidence provided.

- Cite every fact: [SQL] for the query result, [S1], [S2], ... for document excerpts.
- If the evidence does not answer the question, say so plainly. Never guess or use outside
  knowledge.
- Start with the direct answer (numbers with units: INR, %, units), then at most three short
  supporting sentences. When comparing a number with a policy target, state both."""

ANSWER_HUMAN = """Question: {question}

{evidence}"""

def _prompt(system: str, human: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", system), ("human", human)])


router_prompt = _prompt(ROUTER_SYSTEM, ROUTER_HUMAN)
sql_prompt = _prompt(SQL_SYSTEM, SQL_HUMAN)
answer_prompt = _prompt(ANSWER_SYSTEM, ANSWER_HUMAN)

OUT_OF_SCOPE_ANSWER = (
    "I can only answer questions about Kestrel Retail's sales data (stores, products, "
    "promotions, sales and returns) and its business documents (policies, targets and reviews)."
)
NO_EVIDENCE_ANSWER = (
    "I couldn't find information to answer that in the sales data or the business documents."
)

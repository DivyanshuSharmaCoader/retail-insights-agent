"""The LangGraph agent.

Parent graph::

    START -> route --(sql)------> sql_agent ------\\
                   --(docs)-----> retrieve_docs ---> synthesize -> verify -> END
                   --(both)-----> sql_agent + retrieve_docs (parallel, same superstep)
                   --(out_of_scope)-> out_of_scope -> END

``sql_agent`` is a compiled subgraph with a self-correction loop::

    START -> generate_sql -> execute_sql --(error and attempts left)--> generate_sql
                                         --(ok or out of attempts)----> END

When the router picks "both", the SQL subgraph and document retrieval run in parallel in
the same superstep; ``synthesize`` then runs once, after both have written their keys.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from retail_agent.agent import prompts as P
from retail_agent.db.models import metadata
from retail_agent.db.sql_tools import QueryResult, SqlRunner, SQLValidationError, describe_schema
from retail_agent.rag.retriever import Retriever

_CITATION_RE = re.compile(r"\[(SQL|S\d+)\]")


class SQLState(TypedDict, total=False):
    standalone_question: str
    sql: str | None
    sql_explanation: str | None
    sql_result: dict | None
    sql_error: str | None
    sql_attempts: int


class AgentState(SQLState, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    route: str
    route_reason: str
    contexts: list[dict]
    answer: str
    citations: list[str]
    verified: bool
    warnings: list[str]


@dataclass
class AgentDeps:
    llm: BaseChatModel | None
    retriever: Retriever
    sql_runner: SqlRunner
    documents: list[str]
    max_sql_attempts: int = 3
    _schema: str | None = None

    @property
    def schema(self) -> str:
        if self._schema is None:  # built once; includes cached sample rows
            self._schema = describe_schema(self.sql_runner.engine)
        return self._schema


def _history(messages: list[AnyMessage], turns: int = 6) -> str:
    prior = messages[:-1][-turns:]
    if not prior:
        return "(none)"
    return "\n".join(f"{m.type}: {str(m.content)[:400]}" for m in prior)


def _per_turn_reset() -> dict[str, Any]:
    """With a checkpointer, state persists across turns; clear last turn's evidence."""
    return {
        "sql": None, "sql_explanation": None, "sql_result": None, "sql_error": None,
        "sql_attempts": 0, "contexts": [], "answer": "", "citations": [], "verified": False,
        "warnings": [],
    }


def build_sql_subgraph(deps: AgentDeps):
    parser = PydanticOutputParser(pydantic_object=P.SQLDraft)
    chain = P.sql_prompt | deps.llm | parser
    dialect = deps.sql_runner.dialect

    async def generate_sql(state: SQLState) -> dict:
        previous = ""
        if state.get("sql_error"):
            previous = (f"\nPrevious attempt:\n{state.get('sql') or '(unparseable output)'}\n"
                        f"Error: {state['sql_error']}\nWrite a corrected query.")
        try:
            draft = await chain.ainvoke({
                "dialect": dialect,
                "schema": deps.schema,
                "dialect_hints": P.DIALECT_HINTS.get(dialect, ""),
                "format_instructions": parser.get_format_instructions(),
                "question": state["standalone_question"],
                "previous_attempt": previous,
            })
        except (OutputParserException, ValidationError):
            return {"sql": None, "sql_error": "Output was not valid JSON with a 'sql' field."}
        return {"sql": draft.sql, "sql_explanation": draft.explanation}

    async def execute_sql(state: SQLState) -> dict:
        attempts = state.get("sql_attempts", 0) + 1
        if not state.get("sql"):
            return {"sql_attempts": attempts}
        try:
            result = await deps.sql_runner.run(state["sql"])
        except SQLValidationError as exc:
            return {"sql_error": str(exc), "sql_attempts": attempts}
        except TimeoutError:
            return {"sql_error": f"Query timed out after {deps.sql_runner.timeout_s:.0f}s; "
                                 "aggregate more or filter by date.", "sql_attempts": attempts}
        except SQLAlchemyError as exc:
            detail = str(getattr(exc, "orig", exc)).splitlines()[0]
            return {"sql_error": f"Database error: {detail}", "sql_attempts": attempts}
        return {"sql": result.sql, "sql_result": asdict(result), "sql_error": None,
                "sql_attempts": attempts}

    def after_execute(state: SQLState) -> str:
        if state.get("sql_error") and state.get("sql_attempts", 0) < deps.max_sql_attempts:
            return "generate_sql"
        return END

    g = StateGraph(SQLState)
    g.add_node("generate_sql", generate_sql)
    g.add_node("execute_sql", execute_sql)
    g.add_edge(START, "generate_sql")
    g.add_edge("generate_sql", "execute_sql")
    g.add_conditional_edges("execute_sql", after_execute, ["generate_sql", END])
    return g.compile()


def build_graph(deps: AgentDeps, checkpointer=None):
    route_parser = PydanticOutputParser(pydantic_object=P.RouteDecision)
    route_chain = P.router_prompt | deps.llm | route_parser
    answer_chain = P.answer_prompt | deps.llm | StrOutputParser()
    tables = ", ".join(sorted(metadata.tables))

    async def route(state: AgentState) -> dict:
        question = str(state["messages"][-1].content)
        try:
            decision = await route_chain.ainvoke({
                "tables": tables,
                "documents": ", ".join(deps.documents),
                "format_instructions": route_parser.get_format_instructions(),
                "history": _history(state["messages"]),
                "question": question,
            })
        except (OutputParserException, ValidationError):
            # Unparseable router output: gather all evidence rather than fail the request.
            decision = P.RouteDecision(route="both", standalone_question=question,
                                       reason="router output could not be parsed")
        return {**_per_turn_reset(), "question": question, "route": decision.route,
                "route_reason": decision.reason,
                "standalone_question": decision.standalone_question or question}

    def pick_branches(state: AgentState) -> list[str] | str:
        return {
            "sql": ["sql_agent"],
            "docs": ["retrieve_docs"],
            "both": ["sql_agent", "retrieve_docs"],
        }.get(state["route"], "out_of_scope")

    async def retrieve_docs(state: AgentState) -> dict:
        hits = await deps.retriever.aretrieve(state["standalone_question"])
        return {"contexts": [
            {"id": f"S{i}", "source": h.chunk.source, "section": h.chunk.section,
             "text": h.chunk.text, "score": h.score, "dense": h.dense, "sparse": h.sparse}
            for i, h in enumerate(hits, start=1)
        ]}

    async def synthesize(state: AgentState) -> dict:
        parts = []
        result = state.get("sql_result")
        if result:
            table = QueryResult(**{k: v for k, v in result.items() if k != "row_count"})
            parts.append(f"[SQL] query:\n{result['sql']}\n[SQL] result:\n{table.to_markdown()}")
        elif state.get("sql_error"):
            parts.append(f"[SQL] could not be computed: {state['sql_error']}")
        for c in state.get("contexts") or []:
            parts.append(f"[{c['id']}] ({c['source']} > {c['section']})\n{c['text']}")

        has_evidence = bool(result) or bool(state.get("contexts"))
        if not has_evidence:
            answer = P.NO_EVIDENCE_ANSWER
        else:
            answer = await answer_chain.ainvoke({
                "question": state["standalone_question"],
                "evidence": "\n\n".join(parts),
            })
        return {"answer": answer.strip(), "messages": [AIMessage(content=answer.strip())]}

    def verify(state: AgentState) -> dict:
        """Deterministic source control: every citation must point at real evidence."""
        answer = state.get("answer", "")
        cited = list(dict.fromkeys(_CITATION_RE.findall(answer)))
        allowed = {c["id"] for c in state.get("contexts") or []}
        if state.get("sql_result"):
            allowed.add("SQL")
        warnings = []
        unknown = [c for c in cited if c not in allowed]
        if unknown:
            warnings.append(f"Answer cites evidence that was not provided: {', '.join(unknown)}")
        if allowed and not cited and answer != P.NO_EVIDENCE_ANSWER:
            warnings.append("Answer has no citations.")
        if not allowed:
            warnings.append("No supporting evidence was found.")
        if state.get("sql_error") and state.get("route") in ("sql", "both"):
            warnings.append(f"SQL failed after {state.get('sql_attempts', 0)} attempt(s): "
                            f"{state['sql_error']}")
        return {"citations": [c for c in cited if c in allowed], "warnings": warnings,
                "verified": not warnings}

    def out_of_scope(state: AgentState) -> dict:
        return {"answer": P.OUT_OF_SCOPE_ANSWER, "verified": True, "warnings": [],
                "messages": [AIMessage(content=P.OUT_OF_SCOPE_ANSWER)]}

    g = StateGraph(AgentState)
    g.add_node("route", route)
    g.add_node("sql_agent", build_sql_subgraph(deps))
    g.add_node("retrieve_docs", retrieve_docs)
    g.add_node("synthesize", synthesize)
    g.add_node("verify", verify)
    g.add_node("out_of_scope", out_of_scope)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", pick_branches, ["sql_agent", "retrieve_docs", "out_of_scope"])
    g.add_edge("sql_agent", "synthesize")
    g.add_edge("retrieve_docs", "synthesize")
    g.add_edge("synthesize", "verify")
    g.add_edge("verify", END)
    g.add_edge("out_of_scope", END)
    return g.compile(checkpointer=checkpointer)


async def ask(graph, question: str, thread_id: str | None = None, timeout_s: float = 90.0) -> dict:
    """Run one question through a compiled graph and return the final state."""
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
    return await asyncio.wait_for(
        graph.ainvoke({"messages": [HumanMessage(content=question)]}, config=config),
        timeout=timeout_s,
    )

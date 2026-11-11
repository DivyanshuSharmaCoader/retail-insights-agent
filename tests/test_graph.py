from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from retail_agent.agent import prompts as P
from retail_agent.agent.graph import AgentDeps, ask, build_graph
from tests.conftest import DAIRY_MAY_RETURN_RATE_SQL
from tests.fakes import ScriptedChatModel, route, sql

Q_RATE = "What was the Dairy return rate in May 2026?"
Q_POLICY = "At what monthly return rate must Quality Assurance investigate?"
Q_BOTH = "Did the Dairy return rate in May 2026 breach the policy threshold?"


def make_graph(llm, runner, retriever, checkpointer=None):
    deps = AgentDeps(llm=llm, retriever=retriever, sql_runner=runner,
                     documents=sorted({c.source for c in retriever.store.chunks}))
    return build_graph(deps, checkpointer=checkpointer)


async def test_sql_route_answers_from_the_database(runner, retriever) -> None:
    llm = ScriptedChatModel(responses={
        "ROUTE": [route("sql", Q_RATE)],
        "SQL": [sql(DAIRY_MAY_RETURN_RATE_SQL)],
        "ANSWER": ["The Dairy return rate in May 2026 was 6.41% [SQL]."],
    })
    state = await ask(make_graph(llm, runner, retriever), Q_RATE)
    assert state["route"] == "sql"
    assert 5.0 < state["sql_result"]["rows"][0][0] < 8.0
    assert state["contexts"] == []  # retrieval never ran
    assert state["verified"] and state["citations"] == ["SQL"]
    assert "[SQL] result:" in llm.calls[-1][1]  # the answer prompt saw the query result


async def test_sql_self_correction_loop(runner, retriever) -> None:
    llm = ScriptedChatModel(responses={
        "ROUTE": [route("sql", Q_RATE)],
        "SQL": [sql("SELECT * FROM dairy_returns"), sql(DAIRY_MAY_RETURN_RATE_SQL)],
        "ANSWER": ["6.41% [SQL]."],
    })
    state = await ask(make_graph(llm, runner, retriever), Q_RATE)
    assert state["sql_attempts"] == 2
    second_sql_prompt = [h for t, h in llm.calls if t == "SQL"][1]
    assert "Unknown table(s): dairy_returns" in second_sql_prompt  # error fed back to the LLM
    assert state["verified"]


async def test_gives_up_after_max_attempts_without_inventing_an_answer(runner, retriever) -> None:
    llm = ScriptedChatModel(responses={
        "ROUTE": [route("sql", Q_RATE)],
        "SQL": [sql("DROP TABLE sales")] * 3,
    })
    state = await ask(make_graph(llm, runner, retriever), Q_RATE)
    assert state["sql_attempts"] == 3
    assert state["answer"] == P.NO_EVIDENCE_ANSWER
    assert llm.count("ANSWER") == 0  # no evidence, so the LLM is not asked to answer
    assert not state["verified"] and any("SQL failed" in w for w in state["warnings"])


async def test_docs_route_cites_retrieved_excerpts(runner, retriever) -> None:
    llm = ScriptedChatModel(responses={
        "ROUTE": [route("docs", Q_POLICY)],
        "ANSWER": ["Above 3% in any month [S1]."],
    })
    state = await ask(make_graph(llm, runner, retriever), Q_POLICY)
    assert state["route"] == "docs" and state["sql_result"] is None
    assert state["contexts"][0]["source"] == "returns_policy.md"
    assert state["verified"]


async def test_both_route_runs_sql_and_retrieval_then_answers_once(runner, retriever) -> None:
    llm = ScriptedChatModel(responses={
        "ROUTE": [route("both", Q_BOTH)],
        "SQL": [sql(DAIRY_MAY_RETURN_RATE_SQL)],
        "ANSWER": ["Yes: 6.41% [SQL] against a 3% threshold [S1]."],
    })
    state = await ask(make_graph(llm, runner, retriever), Q_BOTH)
    assert state["sql_result"] is not None and state["contexts"]
    assert llm.count("ANSWER") == 1  # fan-in: synthesize runs once after both branches
    assert state["verified"] and set(state["citations"]) == {"SQL", "S1"}


async def test_citation_to_missing_evidence_fails_verification(runner, retriever) -> None:
    llm = ScriptedChatModel(responses={
        "ROUTE": [route("docs", Q_POLICY)],
        "ANSWER": ["Above 3% [S1], see also [S9]."],
    })
    state = await ask(make_graph(llm, runner, retriever), Q_POLICY)
    assert not state["verified"]
    assert any("S9" in w for w in state["warnings"])


async def test_uncited_answer_fails_verification(runner, retriever) -> None:
    llm = ScriptedChatModel(responses={
        "ROUTE": [route("docs", Q_POLICY)],
        "ANSWER": ["Probably around 5%."],
    })
    state = await ask(make_graph(llm, runner, retriever), Q_POLICY)
    assert not state["verified"] and "Answer has no citations." in state["warnings"]


async def test_out_of_scope_skips_tools(runner, retriever) -> None:
    llm = ScriptedChatModel(responses={"ROUTE": [route("out_of_scope", "Tell me a joke")]})
    state = await ask(make_graph(llm, runner, retriever), "Tell me a joke")
    assert state["answer"] == P.OUT_OF_SCOPE_ANSWER
    assert llm.count("SQL") == 0 and llm.count("ANSWER") == 0


async def test_unparseable_router_output_falls_back_to_both(runner, retriever) -> None:
    llm = ScriptedChatModel(responses={
        "ROUTE": ["I think this is about dairy, maybe SQL?"],
        "SQL": [sql(DAIRY_MAY_RETURN_RATE_SQL)],
        "ANSWER": ["6.41% [SQL]."],
    })
    state = await ask(make_graph(llm, runner, retriever), Q_BOTH)
    assert state["route"] == "both"


async def test_follow_up_uses_conversation_memory_and_resets_evidence(runner, retriever) -> None:
    follow_up = "And what was it in June?"
    llm = ScriptedChatModel(responses={
        "ROUTE": [route("both", Q_BOTH),
                  route("sql", "What was the Dairy return rate in June 2026?")],
        "SQL": [sql(DAIRY_MAY_RETURN_RATE_SQL),
                sql(DAIRY_MAY_RETURN_RATE_SQL.replace("-05-", "-06-"))],
        "ANSWER": ["Yes: 6.41% [SQL] vs 3% [S1].", "1.2% in June [SQL]."],
    })
    graph = make_graph(llm, runner, retriever, checkpointer=InMemorySaver())
    await ask(graph, Q_BOTH, thread_id="t1")
    state = await ask(graph, follow_up, thread_id="t1")

    second_route_prompt = [h for t, h in llm.calls if t == "ROUTE"][1]
    assert Q_BOTH in second_route_prompt  # router saw the earlier turn
    assert state["standalone_question"] == "What was the Dairy return rate in June 2026?"
    assert state["contexts"] == []  # turn-1 document evidence did not leak into turn 2
    assert state["sql_result"]["rows"][0][0] < 3.0
    assert len(state["messages"]) == 4  # two human + two AI messages remembered


@pytest.mark.parametrize("bad", ["", "not json at all"])
async def test_unparseable_sql_output_is_retried(runner, retriever, bad) -> None:
    llm = ScriptedChatModel(responses={
        "ROUTE": [route("sql", Q_RATE)],
        "SQL": [bad, sql(DAIRY_MAY_RETURN_RATE_SQL)],
        "ANSWER": ["6.41% [SQL]."],
    })
    state = await ask(make_graph(llm, runner, retriever), Q_RATE)
    assert state["sql_attempts"] == 2 and state["verified"]

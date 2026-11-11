from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from retail_agent.api.main import create_app
from tests.conftest import DAIRY_MAY_RETURN_RATE_SQL
from tests.fakes import ScriptedChatModel, route, sql

Q = "What was the Dairy return rate in May 2026?"


def scripted(n: int = 1) -> ScriptedChatModel:
    return ScriptedChatModel(responses={
        "ROUTE": [route("sql", Q)] * n,
        "SQL": [sql(DAIRY_MAY_RETURN_RATE_SQL)] * n,
        "ANSWER": ["The Dairy return rate in May 2026 was 6.41% [SQL]."] * n,
    })


@pytest.fixture
def client_and_llm(settings):
    llm = scripted(n=3)
    with TestClient(create_app(settings, llm=llm)) as client:
        yield client, llm


def test_health(client_and_llm) -> None:
    client, _ = client_and_llm
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["index_chunks"] > 0 and body["cache"] == "memory"


def test_ask_then_cache_hit(client_and_llm) -> None:
    client, llm = client_and_llm
    first = client.post("/v1/ask", json={"question": Q})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["route"] == "sql" and body["verified"] and not body["cached"]
    assert body["sql_columns"] == ["return_rate_pct"] and body["sql"].endswith("LIMIT 200")
    assert first.headers["x-request-id"]

    rephrased = "  what was the dairy return rate in May 2026"
    second = client.post("/v1/ask", json={"question": rephrased})
    assert second.json()["cached"] is True
    assert llm.count("ROUTE") == 1  # the cached answer did not call the model again


def test_conversation_requests_are_not_cached(client_and_llm) -> None:
    client, llm = client_and_llm
    for _ in range(2):
        r = client.post("/v1/ask", json={"question": Q, "thread_id": "user-1"})
        assert r.status_code == 200 and r.json()["cached"] is False
    assert llm.count("ROUTE") == 2


@pytest.mark.parametrize("payload", [{"question": "hi"}, {"question": Q, "thread_id": "bad id!"}])
def test_validation_errors(client_and_llm, payload) -> None:
    client, _ = client_and_llm
    assert client.post("/v1/ask", json=payload).status_code == 422


def test_stream_emits_steps_then_answer(client_and_llm) -> None:
    client, _ = client_and_llm
    with client.stream("POST", "/v1/ask/stream", json={"question": Q}) as resp:
        text = "".join(resp.iter_text())
    events = [blk for blk in text.split("\n\n") if blk.strip()]
    steps = [json.loads(e.split("data: ", 1)[1])["node"] for e in events if "event: step" in e]
    assert steps[:1] == ["route"]
    assert {"generate_sql", "execute_sql", "synthesize", "verify"} <= set(steps)
    answer = json.loads(events[-1].split("data: ", 1)[1])
    assert "event: answer" in events[-1] and answer["verified"] and "6.41%" in answer["answer"]


def test_starts_without_llm_key_and_returns_503(settings, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["llm_ready"] is False
        r = client.post("/v1/ask", json={"question": Q})
        assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["detail"]

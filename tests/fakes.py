"""A scripted chat model: deterministic LLM responses for graph and API tests."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

_TASK_RE = re.compile(r"TASK: (\w+)")


class ScriptedChatModel(BaseChatModel):
    """Returns queued responses per prompt type (ROUTE / SQL / ANSWER) and records calls."""

    responses: dict[str, list[str]] = Field(default_factory=dict)
    calls: list[tuple[str, str]] = Field(default_factory=list)  # (task, human message)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        system = next((str(m.content) for m in messages if m.type == "system"), "")
        human = next((str(m.content) for m in reversed(messages) if m.type == "human"), "")
        match = _TASK_RE.search(system)
        task = match.group(1) if match else "UNKNOWN"
        self.calls.append((task, human))
        queue = self.responses.get(task) or []
        if not queue:
            raise AssertionError(f"No scripted response left for task {task}")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=queue.pop(0)))])

    def count(self, task: str) -> int:
        return sum(1 for t, _ in self.calls if t == task)


def route(route: str, question: str, reason: str = "test") -> str:
    import json

    return json.dumps({"route": route, "standalone_question": question, "reason": reason})


def sql(query: str, explanation: str = "test query") -> str:
    import json

    return "```json\n" + json.dumps({"sql": query, "explanation": explanation}) + "\n```"

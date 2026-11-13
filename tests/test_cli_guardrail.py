from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import cli


def test_guardrail_history_keeps_recent_user_and_assistant_turns() -> None:
    messages = [
        HumanMessage(content="¿Puedo pedir la pensión de viudedad?"),
        AIMessage(content="Depende de si está casado."),
        ToolMessage(content="resultado irrelevante", tool_call_id="call-1"),
        HumanMessage(content="   "),
        AIMessage(content="¿Está casado?"),
    ]

    assert cli._guardrail_history(messages, 3) == [
        {"role": "user", "content": "¿Puedo pedir la pensión de viudedad?"},
        {"role": "assistant", "content": "Depende de si está casado."},
        {"role": "assistant", "content": "¿Está casado?"},
    ]


def test_guardrail_history_limits_the_number_of_turns() -> None:
    messages = [
        HumanMessage(content="u1"),
        AIMessage(content="a1"),
        HumanMessage(content="u2"),
        AIMessage(content="a2"),
    ]

    assert cli._guardrail_history(messages, 1) == [
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert cli._guardrail_history(messages, 0) == []


def test_guardrail_passes_history_to_check_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    monkeypatch.setattr("guardrails.enabled", lambda: True)

    def fake_check_input(query, config=None, history=None):
        captured["query"] = query
        captured["history"] = history
        return SimpleNamespace(allowed=True)

    monkeypatch.setattr("guardrails.check_input", fake_check_input)

    result = cli._guardrail(
        "Sí.",
        [HumanMessage(content="¿Está casado?"), AIMessage(content="Depende.")],
    )

    assert result.allowed is True
    assert captured == {
        "query": "Sí.",
        "history": [
            {"role": "user", "content": "¿Está casado?"},
            {"role": "assistant", "content": "Depende."},
        ],
    }

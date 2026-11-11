from __future__ import annotations

from dataclasses import replace

import pytest

from agent.agent import build_model
from agent.config import Settings


def test_invalid_api_mode_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_API_MODE", "bogus")

    with pytest.raises(ValueError, match="AGENT_API_MODE"):
        Settings.from_env()


def test_responses_mode_is_stateless(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_API_MODE", "responses")
    settings = replace(Settings.from_env(), api_key="dummy")

    model = build_model(settings)

    assert model.use_responses_api is True
    assert model.use_previous_response_id is False
    assert model.include == ["reasoning.encrypted_content"]


def test_chat_completions_mode_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_API_MODE", "chat_completions")
    settings = replace(Settings.from_env(), api_key="dummy")

    model = build_model(settings)

    assert model.use_responses_api is False
    assert model.include is None


def test_empty_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_DEVICE", "")

    assert Settings.from_env().device == "cpu"

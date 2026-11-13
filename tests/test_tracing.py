from __future__ import annotations

import pytest

from agent.tracing import (
    last_trace_id,
    span,
    trace_config,
    trace_root,
    trace_url,
    tracing_enabled,
)


class _Handler:
    last_trace_id = "abc123"


def test_tracing_disabled_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert tracing_enabled() is False


def test_tracing_can_be_disabled_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")

    assert tracing_enabled() is False


def test_tracing_enabled_with_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")

    assert tracing_enabled() is True


def test_trace_url_uses_project_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:3000/")
    monkeypatch.setenv("LANGFUSE_PROJECT_ID", "local-project")

    assert trace_url("abc123") == "http://localhost:3000/project/local-project/traces/abc123"


def test_trace_url_is_none_without_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PROJECT_ID", raising=False)

    assert trace_url("abc123") is None


def test_last_trace_id_reads_the_callback_handler() -> None:
    assert last_trace_id({"callbacks": [_Handler()]}) == "abc123"
    assert last_trace_id({}) is None


def test_trace_root_is_a_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")

    with trace_root("test", session_id="session-1") as trace_context:
        assert trace_context is None


def test_trace_config_is_empty_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")

    assert trace_config(session_id="session-1", trace_name="test") == {}


def test_guardrail_span_is_a_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")

    with span("input_guardrail", as_type="guardrail") as observation:
        assert observation is None

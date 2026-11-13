"""Optional Langfuse tracing for CLI runs.

Tracing is enabled only when both Langfuse keys are present. The SDK is imported
lazily so commands that do not trace (or a machine without Langfuse) do not pay
the OpenTelemetry import cost.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from dotenv import load_dotenv

from agent.config import ROOT

load_dotenv(ROOT / ".env")


def tracing_enabled() -> bool:
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() in {"0", "false", "no", "off"}:
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@contextmanager
def span(
    name: str,
    *,
    as_type: str = "span",
    **kwargs: Any,
) -> Iterator[Any]:
    """Nest a span under the current Langfuse observation, if tracing is on."""
    if not tracing_enabled():
        yield None
        return

    from langfuse import get_client

    with get_client().start_as_current_observation(
        as_type=as_type,
        name=name,
        **kwargs,
    ) as observation:
        yield observation


@contextmanager
def trace_root(
    name: str,
    *,
    session_id: str,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    input_data: Any | None = None,
) -> Iterator[dict[str, str] | None]:
    """Start a root span and return the context needed to join its trace.

    The guardrail and the LangChain agent run inside this context, so both end up
    in the same trace. Pass the yielded dictionary to :func:`trace_config` as
    ``trace_context`` so the LangChain callback nests under this root span.
    """
    if not tracing_enabled():
        yield None
        return

    from langfuse import get_client, propagate_attributes

    client = get_client()
    with (
        client.start_as_current_observation(
            as_type="span",
            name=name,
            input=input_data,
        ),
        propagate_attributes(
            trace_name=name,
            session_id=session_id,
            tags=tags or [],
            metadata=metadata or {},
        ),
    ):
        trace_id = client.get_current_trace_id()
        parent_span_id = client.get_current_observation_id()
        if trace_id and parent_span_id:
            yield {"trace_id": trace_id, "parent_span_id": parent_span_id}
        else:
            yield None


def last_trace_id(config: dict[str, Any]) -> str | None:
    callbacks = config.get("callbacks") or []
    if not callbacks:
        return None
    return getattr(callbacks[0], "last_trace_id", None)


def trace_url(trace_id: str) -> str | None:
    base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
    project_id = os.getenv("LANGFUSE_PROJECT_ID")
    if not base_url or not project_id or not trace_id:
        return None
    return f"{base_url.rstrip('/')}/project/{project_id}/traces/{trace_id}"


def trace_config(
    session_id: str,
    trace_name: str,
    tags: list[str] | None = None,
    trace_context: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a LangChain config that sends this run to Langfuse."""
    if not tracing_enabled():
        return {}

    from langfuse.langchain import CallbackHandler

    metadata: dict[str, Any] = {
        "langfuse_session_id": session_id,
        "langfuse_trace_name": trace_name,
    }
    if tags:
        metadata["langfuse_tags"] = tags
    return {
        "callbacks": [CallbackHandler(trace_context=trace_context)],
        "metadata": metadata,
    }


def flush() -> None:
    """Send buffered spans before a short-lived CLI process exits."""
    if not tracing_enabled():
        return
    from langfuse import get_client

    try:
        get_client().flush()
    except Exception:  # noqa: BLE001 - tracing must never break the CLI
        import logging

        logging.getLogger(__name__).warning("Could not flush Langfuse traces.")

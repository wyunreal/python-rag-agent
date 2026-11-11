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
def span(name: str, **kwargs: Any) -> Iterator[Any]:
    """Nest a span under the current Langfuse observation, if tracing is on."""
    if not tracing_enabled():
        yield None
        return

    from langfuse import get_client

    with get_client().start_as_current_observation(
        as_type="span",
        name=name,
        **kwargs,
    ) as observation:
        yield observation


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
    return {"callbacks": [CallbackHandler()], "metadata": metadata}


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

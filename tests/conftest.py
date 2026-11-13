from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def disable_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests deterministic and offline: never create Langfuse spans."""
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")

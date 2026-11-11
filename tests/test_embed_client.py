from __future__ import annotations

import json
from urllib.error import URLError

import pytest

from agent.config import Settings
from agent.index import _cached_server_vector, check_embed_server, embed_query


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps({"vectors": [[0.0] * 1024]}).encode()


def test_embed_query_uses_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return _Response()

    monkeypatch.setenv("AGENT_EMBED_SERVER", "http://embed.test")
    monkeypatch.setattr("agent.index.urllib.request.urlopen", fake_urlopen)
    _cached_server_vector.cache_clear()

    vector = embed_query(Settings.from_env(), "consulta de prueba")

    assert vector.shape == (1024,)
    assert calls == ["http://embed.test/embed"]


def test_check_embed_server_fails_when_down(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_urlopen(*args, **kwargs):
        raise URLError("connection refused")

    monkeypatch.setenv("AGENT_EMBED_SERVER", "http://embed.test")
    monkeypatch.setattr("agent.index.urllib.request.urlopen", failing_urlopen)

    with pytest.raises(RuntimeError, match="no responde"):
        check_embed_server()


def test_check_embed_server_skips_without_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_EMBED_SERVER", raising=False)

    check_embed_server()


def test_embed_query_does_not_fall_back_when_server_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_urlopen(*args, **kwargs):
        raise URLError("connection refused")

    monkeypatch.setenv("AGENT_EMBED_SERVER", "http://embed.test")
    monkeypatch.setattr("agent.index.urllib.request.urlopen", failing_urlopen)
    _cached_server_vector.cache_clear()

    with pytest.raises(RuntimeError, match="no responde"):
        embed_query(Settings.from_env(), "otra consulta distinta")

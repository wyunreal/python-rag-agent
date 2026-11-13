from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from decision import DecisionBackendError, DecisionRequest, SystemOneBackend


def request(name: str = "q") -> DecisionRequest:
    return DecisionRequest(
        model="clef:27b",
        context="¿Es delito robar?",
        question="Is this a legal question?",
        name=name,
    )


def backend(handler: Any, **kwargs: Any) -> SystemOneBackend:
    return SystemOneBackend(
        base_url="https://ollama.com/v1",
        api_key="secret",
        transport=httpx.MockTransport(handler),
        retry_backoff=0,
        **kwargs,
    )


def test_payload_url_and_auth_header() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["auth"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"answers": {"q": {"type": "noul", "noul": 0.75}}},
        )

    with backend(handler) as client:
        probability = client.judge(request())

    assert probability == 75.0
    assert captured == {
        "method": "POST",
        "url": "https://ollama.com/v1/systemone",
        "auth": "Bearer secret",
        "body": {
            "model": "clef:27b",
            "state": "¿Es delito robar?",
            "questions": {
                "q": {
                    "type": "noul",
                    "instructions": "Is this a legal question?",
                }
            },
        },
    }


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://ollama.com/v1", "https://ollama.com/v1/systemone"),
        ("https://ollama.com/v1/", "https://ollama.com/v1/systemone"),
        ("https://ollama.com", "https://ollama.com/v1/systemone"),
        ("https://ollama.com/v1/systemone", "https://ollama.com/v1/systemone"),
        ("http://localhost:11434", "http://localhost:11434/v1/systemone"),
    ],
)
def test_systemone_url_normalization(base_url: str, expected: str) -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"answers": {"q": {"noul": 0.5}}})

    with SystemOneBackend(
        base_url=base_url,
        api_key="secret",
        transport=httpx.MockTransport(handler),
        retry_backoff=0,
    ) as client:
        client.judge(request())

    assert captured["url"] == expected


def test_client_error_is_not_retried() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": "not a decision model"})

    with backend(handler) as client, pytest.raises(DecisionBackendError, match="HTTP 400"):
        client.judge(request())

    assert calls == 1


def test_server_error_is_retried() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500, text="temporary failure")
        return httpx.Response(200, json={"answers": {"q": {"noul": 0.5}}})

    with backend(handler, max_attempts=2) as client:
        assert client.judge(request()) == 50.0

    assert calls == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"answers": {}},
        {"answers": {"q": {"noul": 1.2}}},
        {"answers": {"q": {"noul": "0.5"}}},
        {"answers": {"q": {"noul": 0.5, "type": "choice"}}},
    ],
)
def test_invalid_responses_are_rejected(payload: dict[str, Any]) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with backend(handler) as client, pytest.raises(DecisionBackendError):
        client.judge(request())

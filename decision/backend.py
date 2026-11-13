"""HTTP backend for Ollama's System One (Jev-style) decision API.

The endpoint is ``POST {base_url}/systemone``. A request contains a ``state``
and a map of named questions; a ``noul`` question returns the probability that
the answer is yes. This module turns that probability into a percentage.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Protocol

import httpx
from dotenv import load_dotenv

from decision.types import DecisionRequest

DEFAULT_BASE_URL = "https://ollama.com/v1"


class DecisionBackendError(RuntimeError):
    """The decision backend could not produce a valid probability."""


class JudgeBackend(Protocol):
    """Anything that can answer one yes/no question with ``P(yes)``."""

    def judge(self, request: DecisionRequest) -> float:
        """Return the probability of yes as a percentage in ``[0, 100]``."""


def _systemone_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise ValueError("base_url must not be empty")
    if base.endswith("/systemone"):
        return base
    if base.endswith("/v1"):
        return f"{base}/systemone"
    return f"{base}/v1/systemone"


def _response_detail(response: httpx.Response) -> str:
    text = response.text.strip()
    return text[:300] if text else response.reason_phrase


def _parse_noul(data: Any, name: str) -> float:
    try:
        answer = data["answers"][name]
    except (KeyError, TypeError) as exc:
        raise DecisionBackendError(
            f"Unexpected System One response, missing answers[{name!r}]: {data!r}"
        ) from exc

    if not isinstance(answer, dict):
        raise DecisionBackendError(
            f"System One returned a non-object answer for {name!r}: {answer!r}"
        )
    answer_type = answer.get("type")
    if answer_type is not None and answer_type != "noul":
        raise DecisionBackendError(
            f"System One returned type {answer_type!r} instead of 'noul' for {name!r}"
        )

    value = answer.get("noul")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionBackendError(
            f"System One returned a non-numeric noul for {name!r}: {value!r}"
        )
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise DecisionBackendError(
            f"System One returned noul outside [0, 1] for {name!r}: {value!r}"
        )
    return probability


class SystemOneBackend:
    """Parallel-safe client for ``/v1/systemone``.

    ``httpx.Client`` is thread-safe, so one instance can serve every request of
    one :func:`decision.engine.run_judges` call.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        max_attempts: int = 2,
        retry_backoff: float = 0.25,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if retry_backoff < 0:
            raise ValueError("retry_backoff must not be negative")

        self._url = _systemone_url(base_url)
        self._api_key = api_key
        self._max_attempts = max_attempts
        self._retry_backoff = retry_backoff
        self._client = httpx.Client(timeout=timeout, transport=transport)

    @classmethod
    def from_env(cls, *, dotenv_path: str | Path | None = None) -> SystemOneBackend:
        """Build a backend from ``OLLAMA_API_KEY`` and ``OLLAMA_BASE_URL``."""
        load_dotenv(dotenv_path)
        api_key = os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_CLOUD_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Falta OLLAMA_API_KEY. Crea una clave en "
                "https://ollama.com/settings/keys y añádela a tu .env."
            )
        return cls(
            base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
            api_key=api_key,
        )

    def judge(self, request: DecisionRequest) -> float:
        """Send one ``noul`` question and return ``P(yes) * 100``."""
        payload = {
            "model": request.model,
            "state": request.context,
            "questions": {
                request.name: {
                    "type": "noul",
                    "instructions": request.question,
                }
            },
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                response = self._client.post(self._url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_error = exc
            else:
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    raise DecisionBackendError(
                        "System One rejected the request "
                        f"(HTTP {response.status_code}): {_response_detail(response)}"
                    )
                if response.status_code >= 400:
                    last_error = DecisionBackendError(
                        "System One returned "
                        f"HTTP {response.status_code}: {_response_detail(response)}"
                    )
                else:
                    try:
                        data = response.json()
                    except ValueError as exc:
                        raise DecisionBackendError("System One returned invalid JSON.") from exc
                    return round(_parse_noul(data, request.name) * 100.0, 4)

            if attempt + 1 < self._max_attempts and self._retry_backoff:
                time.sleep(self._retry_backoff * (attempt + 1))

        raise DecisionBackendError(
            f"System One request failed after {self._max_attempts} attempt(s): {last_error}"
        ) from last_error

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SystemOneBackend:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

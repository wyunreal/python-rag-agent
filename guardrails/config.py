"""Configuration for the input guardrail."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from agent.config import ROOT
from decision.types import validate_threshold

DEFAULT_MODEL = "clef:27b"
DEFAULT_THRESHOLD = 70.0
DEFAULT_JUDGES = 1
DEFAULT_HISTORY_TURNS = 3


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True, slots=True)
class InputGuardConfig:
    """Decision model, threshold, requests per check and history window.

    ``judges`` is the number of requests sent for each of the two checks. With
    only one decision model available, the same question is repeated; when more
    models are added, the caller can build its own requests instead.

    ``history_turns`` is the number of previous conversation turns sent as
    context in chat mode. A turn is a user message and the assistant response
    that follows it. Zero disables the context.
    """

    model: str = DEFAULT_MODEL
    threshold: float = DEFAULT_THRESHOLD
    judges: int = DEFAULT_JUDGES
    history_turns: int = DEFAULT_HISTORY_TURNS

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("GUARDRAIL_MODEL must not be empty")
        object.__setattr__(self, "threshold", validate_threshold(self.threshold))
        if isinstance(self.judges, bool) or not isinstance(self.judges, int):
            raise TypeError("GUARDRAIL_JUDGES must be an integer")
        if self.judges < 1:
            raise ValueError("GUARDRAIL_JUDGES must be at least 1")
        if isinstance(self.history_turns, bool) or not isinstance(self.history_turns, int):
            raise TypeError("GUARDRAIL_HISTORY_TURNS must be an integer")
        if self.history_turns < 0:
            raise ValueError("GUARDRAIL_HISTORY_TURNS must be zero or greater")

    @classmethod
    def from_env(cls) -> InputGuardConfig:
        load_dotenv(ROOT / ".env")
        return cls(
            model=os.getenv("GUARDRAIL_MODEL", DEFAULT_MODEL),
            threshold=_float("GUARDRAIL_THRESHOLD", DEFAULT_THRESHOLD),
            judges=_int("GUARDRAIL_JUDGES", DEFAULT_JUDGES),
            history_turns=_int("GUARDRAIL_HISTORY_TURNS", DEFAULT_HISTORY_TURNS),
        )


def enabled() -> bool:
    """Return whether the input guardrail is enabled (default: yes)."""
    load_dotenv(ROOT / ".env")
    raw = os.getenv("AGENT_INPUT_GUARDRAIL", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}

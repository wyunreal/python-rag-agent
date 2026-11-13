"""Core data types for the generic decision engine.

A :class:`DecisionRequest` is one yes/no question sent to one decision model.
The model must return the probability of *yes* (``noul`` in the System One
API). The engine converts that probability to a percentage in ``[0, 100]`` and
aggregates a list of requests into a :class:`DecisionResult`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")

Context = str | dict[str, Any] | list[Any]


def validate_threshold(value: float) -> float:
    """Return ``value`` as a float in ``(0, 100]`` or raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("threshold must be a number")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 100.0:
        raise ValueError("threshold must be greater than 0 and at most 100")
    return threshold


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """One model, one context and one yes/no question.

    Attributes:
        model: decision model name, for example ``"clef:27b"``.
        context: the System One ``state``. It may be text or JSON data.
        question: the yes/no instruction. ``P(yes)`` is the confidence that the
            desired condition holds.
        name: question name used by the System One response. It must be unique
            inside one HTTP request; here every request is a separate HTTP call,
            so it only needs to be a valid identifier.
    """

    model: str
    context: Context
    question: str
    name: str = "decision"

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must not be empty")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must not be empty")
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ValueError("name must contain 1-100 ASCII letters, digits, '_', '.' or '-'")
        if isinstance(self.context, str):
            if not self.context.strip():
                raise ValueError("context must not be empty")
        elif isinstance(self.context, (dict, list)):
            if not self.context:
                raise ValueError("context must not be empty")
        else:
            raise TypeError("context must be a non-empty string, dict or list")


@dataclass(frozen=True, slots=True)
class DecisionConfig:
    """Configuration for one aggregation.

    ``threshold`` is a percentage: a judge counts as *definitive* only when its
    certainty, ``max(p, 100 - p)``, is at least this value.
    """

    threshold: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold", validate_threshold(self.threshold))


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """Outcome of one decision request."""

    request: DecisionRequest
    probability: float | None = None
    vote: bool | None = None
    certainty: float | None = None
    definitive: bool = False
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None and self.probability is not None


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Aggregated decision.

    Attributes:
        decision: final boolean. ``True`` means yes/allow, ``False`` means
            no/block. Ties and a lack of definitive votes resolve to ``False``.
        certainty: consolidated percentage of certainty in ``decision``.
        support: number of definitive requests that voted for ``decision``.
        threshold: threshold used for this aggregation.
        definitive: number of definitive judges.
        valid: number of requests that produced a valid probability.
        judges: per-request outcome, in the same order as the input.
    """

    decision: bool
    certainty: float
    support: int
    threshold: float
    definitive: int
    valid: int
    judges: tuple[JudgeResult, ...]

    @property
    def total(self) -> int:
        return len(self.judges)

    @property
    def failed(self) -> int:
        return self.total - self.valid

    @property
    def abstained(self) -> int:
        return self.valid - self.definitive

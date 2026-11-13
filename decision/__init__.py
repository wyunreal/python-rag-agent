"""Generic Jev-style decision engine for Ollama's System One API.

Public contract::

    from decision import DecisionConfig, DecisionRequest, decide

    result = decide(
        [DecisionRequest(model="clef:27b", context=text, question=question)],
        DecisionConfig(threshold=70),
    )

``result.decision`` is the final boolean, ``result.certainty`` the consolidated
percentage and ``result.support`` the number of requests backing the decision.
"""

from decision.backend import (
    DecisionBackendError,
    JudgeBackend,
    SystemOneBackend,
)
from decision.engine import aggregate, decide, run_judges
from decision.types import (
    DecisionConfig,
    DecisionRequest,
    DecisionResult,
    JudgeResult,
    validate_threshold,
)

__all__ = [
    "DecisionBackendError",
    "DecisionConfig",
    "DecisionRequest",
    "DecisionResult",
    "JudgeBackend",
    "JudgeResult",
    "SystemOneBackend",
    "aggregate",
    "decide",
    "run_judges",
    "validate_threshold",
]

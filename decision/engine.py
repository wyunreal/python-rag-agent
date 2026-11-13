"""Parallel execution and aggregation for the generic decision engine.

The public entry point is :func:`decide`::

    result = decide(requests, DecisionConfig(threshold=70))

Each request is sent in parallel. A request whose backend fails is recorded
with an error and excluded from the vote. If no judge reaches the threshold, or
if the definitive votes tie, the final decision is ``False`` (fail-closed).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

from decision.backend import JudgeBackend, SystemOneBackend
from decision.types import DecisionConfig, DecisionRequest, DecisionResult, JudgeResult


def _judge_result(request: DecisionRequest, probability: float) -> JudgeResult:
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise TypeError("judge probability must be a number")
    value = float(probability)
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise ValueError("judge probability must be between 0 and 100")

    if value > 50.0:
        vote: bool | None = True
    elif value < 50.0:
        vote = False
    else:
        # 50/50 carries no side; it is neither a yes nor a no.
        vote = None

    return JudgeResult(
        request=request,
        probability=value,
        vote=vote,
        certainty=max(value, 100.0 - value),
    )


def run_judges(
    requests: Sequence[DecisionRequest],
    *,
    backend: JudgeBackend | None = None,
) -> tuple[JudgeResult, ...]:
    """Send every request in parallel and return one result per request.

    A custom ``backend`` is useful for tests and for other decision providers.
    When it is omitted, a :class:`SystemOneBackend` is built from the
    environment and closed before returning.
    """
    requests = tuple(requests)
    if not requests:
        return ()

    owns_backend = backend is None
    if backend is None:
        backend = SystemOneBackend.from_env()

    try:
        results: list[JudgeResult | None] = [None] * len(requests)
        with ThreadPoolExecutor(max_workers=len(requests)) as executor:
            futures = {
                executor.submit(backend.judge, request): index
                for index, request in enumerate(requests)
            }
            for future in as_completed(futures):
                index = futures[future]
                request = requests[index]
                try:
                    probability = future.result()
                    results[index] = _judge_result(request, probability)
                except Exception as exc:  # noqa: BLE001 - one judge must not abort the rest
                    results[index] = JudgeResult(
                        request=request,
                        error=f"{type(exc).__name__}: {exc}",
                    )
        return tuple(result for result in results if result is not None)
    finally:
        if owns_backend and isinstance(backend, SystemOneBackend):
            backend.close()


def _certainty_for(judge: JudgeResult, decision: bool) -> float:
    if judge.probability is None:
        raise ValueError("definitive judge without probability")
    return judge.probability if decision else 100.0 - judge.probability


def aggregate(
    judges: Sequence[JudgeResult],
    config: DecisionConfig,
) -> DecisionResult:
    """Aggregate judges into one decision.

    A judge is definitive when it has a side (``p != 50``) and its certainty
    ``max(p, 100 - p)`` is at least ``config.threshold``. The final decision is
    the majority of definitive votes; a tie blocks. ``certainty`` is the mean
    probability that the definitive judges assign to the winning answer, so a
    dissenting judge lowers it. ``support`` counts definitive votes for the
    winner.
    """
    enriched = tuple(
        replace(
            judge,
            definitive=(
                judge.probability is not None
                and judge.vote is not None
                and judge.certainty is not None
                and judge.certainty >= config.threshold
            ),
        )
        for judge in judges
    )
    valid = sum(1 for judge in enriched if judge.probability is not None)
    definitive = tuple(judge for judge in enriched if judge.definitive)

    if not definitive:
        return DecisionResult(
            decision=False,
            certainty=0.0,
            support=0,
            threshold=config.threshold,
            definitive=0,
            valid=valid,
            judges=enriched,
        )

    yes = sum(1 for judge in definitive if judge.vote is True)
    no = sum(1 for judge in definitive if judge.vote is False)
    decision = yes > no
    support = yes if decision else no
    certainty = sum(_certainty_for(judge, decision) for judge in definitive) / len(definitive)

    return DecisionResult(
        decision=decision,
        certainty=round(certainty, 2),
        support=support,
        threshold=config.threshold,
        definitive=len(definitive),
        valid=valid,
        judges=enriched,
    )


def decide(
    requests: Sequence[DecisionRequest],
    config: DecisionConfig,
    *,
    backend: JudgeBackend | None = None,
) -> DecisionResult:
    """Run every request in parallel and aggregate the result."""
    requests = tuple(requests)
    if not requests:
        raise ValueError("at least one DecisionRequest is required")
    return aggregate(run_judges(requests, backend=backend), config)

"""Input guardrail built on the generic decision engine.

Two independent questions are asked for every query:

1. Is this a legal consultation?
2. Does it avoid a first-person admission of a potentially criminal fact (and
   avoid asking for help to commit or conceal a crime)?

Each question is sent to the decision model ``judges`` times, in parallel. The
query is allowed only when both checks pass. The state includes the recent
conversation when the caller provides it, so short follow-ups are evaluated in
context.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from decision import (
    DecisionConfig,
    DecisionRequest,
    DecisionResult,
    JudgeBackend,
    JudgeResult,
    aggregate,
    run_judges,
)
from guardrails.config import InputGuardConfig
from guardrails.policy import (
    CHECKS,
    LEGAL_DOMAIN,
    NO_SELF_INCRIMINATION,
)


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    """Final guardrail verdict with the breakdown of both checks."""

    allowed: bool
    certainty: float
    support: int
    checks: Mapping[str, DecisionResult]
    blocked_by: tuple[str, ...]
    reason: str


def _state(
    query: str,
    history: Sequence[Mapping[str, str]] | None,
) -> dict[str, Any]:
    conversation: list[dict[str, str]] = []
    for message in history or ():
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        conversation.append({"role": role, "content": content})
    return {"conversation": conversation, "message": query}


def _requests(
    query: str,
    config: InputGuardConfig,
    history: Sequence[Mapping[str, str]] | None,
) -> list[DecisionRequest]:
    state = _state(query, history)
    requests: list[DecisionRequest] = []
    for index in range(config.judges):
        for check_name, question in CHECKS.items():
            requests.append(
                DecisionRequest(
                    model=config.model,
                    context=state,
                    question=question,
                    name=f"{check_name}_{index}",
                )
            )
    return requests


def _judges_for(
    judges: Sequence[JudgeResult],
    check_name: str,
) -> tuple[JudgeResult, ...]:
    prefix = f"{check_name}_"
    return tuple(judge for judge in judges if judge.request.name.startswith(prefix))


def _reason(
    blocked_by: Sequence[str],
    checks: Mapping[str, DecisionResult],
) -> str:
    if not blocked_by:
        return (
            "La consulta es de índole jurídica y no contiene una admisión "
            "incriminatoria en primera persona."
        )

    parts: list[str] = []
    if LEGAL_DOMAIN in blocked_by:
        if checks[LEGAL_DOMAIN].definitive == 0:
            parts.append(
                "no hay consenso suficiente para confirmar que sea una consulta de índole jurídica"
            )
        else:
            parts.append("no parece una consulta de índole jurídica")

    if NO_SELF_INCRIMINATION in blocked_by:
        if checks[NO_SELF_INCRIMINATION].definitive == 0:
            parts.append(
                "no hay consenso suficiente para descartar una admisión incriminatoria "
                "en primera persona"
            )
        else:
            parts.append(
                "contiene una admisión o confesión en primera persona de un hecho "
                "potencialmente delictivo, o pide ayuda para cometerlo o encubrirlo"
            )

    message = "No puedo atender esta consulta: " + "; ".join(parts) + "."
    if NO_SELF_INCRIMINATION in blocked_by:
        message += (
            " Puedo ayudarte con una pregunta general sobre el delito o con tus "
            "derechos como víctima."
        )
    return message


def check_input(
    query: str,
    config: InputGuardConfig | None = None,
    *,
    history: Sequence[Mapping[str, str]] | None = None,
    backend: JudgeBackend | None = None,
) -> GuardrailResult:
    """Classify a user query before it reaches the RAG agent.

    Args:
        query: raw user message.
        config: guardrail configuration. Defaults to ``InputGuardConfig.from_env()``.
        history: previous conversation turns as ``{"role", "content"}`` mappings.
            They are sent as context so short follow-ups such as "Sí" are not
            mistaken for unrelated messages.
        backend: optional decision backend, mainly for tests.

    Returns:
        A :class:`GuardrailResult`. On any backend failure the affected judge is
        discarded; if no judge is definitive, the query is blocked.
    """
    config = config or InputGuardConfig.from_env()
    normalized = query.strip()
    if not normalized:
        return GuardrailResult(
            allowed=False,
            certainty=100.0,
            support=0,
            checks={},
            blocked_by=("empty",),
            reason="No puedo atender esta consulta: está vacía.",
        )

    decision_config = DecisionConfig(threshold=config.threshold)
    judges = run_judges(
        _requests(normalized, config, history),
        backend=backend,
    )
    checks = {
        check_name: aggregate(_judges_for(judges, check_name), decision_config)
        for check_name in CHECKS
    }

    allowed = all(check.decision for check in checks.values())
    blocked_by = tuple(check_name for check_name, check in checks.items() if not check.decision)

    if allowed:
        certainty = min(check.certainty for check in checks.values())
        support = sum(check.support for check in checks.values())
    else:
        blocking = [checks[check_name] for check_name in blocked_by]
        certainty = max(check.certainty for check in blocking)
        support = sum(check.support for check in blocking)

    return GuardrailResult(
        allowed=allowed,
        certainty=round(certainty, 2),
        support=support,
        checks=checks,
        blocked_by=blocked_by,
        reason=_reason(blocked_by, checks),
    )

from __future__ import annotations

import threading

import pytest

from guardrails import InputGuardConfig, check_input, enabled
from guardrails.input_guard import decision_payload
from guardrails.policy import (
    LEGAL_DOMAIN_QUESTION,
    NO_SELF_INCRIMINATION_QUESTION,
)


class QuestionBackend:
    """Return a fixed probability depending on which check is being asked."""

    def __init__(
        self,
        legal: float | Exception,
        incrimination: float | Exception,
    ) -> None:
        self.legal = legal
        self.incrimination = incrimination
        self.requests: list = []
        self._lock = threading.Lock()

    def judge(self, request):
        with self._lock:
            self.requests.append(request)
        value = self.legal if request.name.startswith("legal_") else self.incrimination
        if isinstance(value, Exception):
            raise value
        return value


def config(**overrides) -> InputGuardConfig:
    values = {"threshold": 70.0, "judges": 1}
    values.update(overrides)
    return InputGuardConfig(**values)


def test_both_checks_pass() -> None:
    backend = QuestionBackend(95, 95)

    result = check_input(
        "¿Cuál es el plazo máximo de detención preventiva?",
        config(),
        backend=backend,
    )

    assert result.allowed is True
    assert result.certainty == 95.0
    assert result.support == 2
    assert result.blocked_by == ()
    assert set(result.checks) == {"legal", "no_self_incrimination"}
    assert len(backend.requests) == 2
    assert {request.question for request in backend.requests} == {
        LEGAL_DOMAIN_QUESTION,
        NO_SELF_INCRIMINATION_QUESTION,
    }
    assert {request.context["message"] for request in backend.requests} == {
        "¿Cuál es el plazo máximo de detención preventiva?"
    }
    assert all(request.context["conversation"] == [] for request in backend.requests)


def test_decision_payload_contains_trace_details() -> None:
    result = check_input(
        "¿Qué pena tiene robar?",
        config(),
        backend=QuestionBackend(95, 90),
    )

    payload = decision_payload(result.checks["legal"])

    assert payload["decision"] is True
    assert payload["certainty"] == 95.0
    assert payload["judges"][0]["name"] == "legal_0"
    assert payload["judges"][0]["probability"] == 95.0
    assert payload["judges"][0]["definitive"] is True


def test_history_is_included_and_filtered() -> None:
    history = [
        {"role": "user", "content": "¿Puedo pedir la pensión de viudedad?"},
        {"role": "assistant", "content": "Depende de si está casado."},
        {"role": "tool", "content": "resultado irrelevante"},
        {"role": "user", "content": "   "},
    ]
    backend = QuestionBackend(95, 95)

    result = check_input(
        "No, no estoy casado.",
        config(),
        history=history,
        backend=backend,
    )

    assert result.allowed is True
    for request in backend.requests:
        assert request.context == {
            "conversation": [
                {"role": "user", "content": "¿Puedo pedir la pensión de viudedad?"},
                {"role": "assistant", "content": "Depende de si está casado."},
            ],
            "message": "No, no estoy casado.",
        }


def test_history_is_trimmed_to_configured_turns() -> None:
    history = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]
    backend = QuestionBackend(95, 95)

    result = check_input(
        "Sí.",
        config(history_turns=2),
        history=history,
        backend=backend,
    )

    assert result.allowed is True
    expected = [
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]
    for request in backend.requests:
        assert request.context["conversation"] == expected


def test_history_can_be_disabled() -> None:
    history = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    backend = QuestionBackend(95, 95)

    result = check_input(
        "Sí.",
        config(history_turns=0),
        history=history,
        backend=backend,
    )

    assert result.allowed is True
    for request in backend.requests:
        assert request.context["conversation"] == []


def test_non_legal_query_is_blocked() -> None:
    result = check_input(
        "¿Cómo se hace una tortilla?",
        config(),
        backend=QuestionBackend(5, 95),
    )

    assert result.allowed is False
    assert result.blocked_by == ("legal",)
    assert result.certainty == 95.0
    assert result.support == 1
    assert "índole jurídica" in result.reason


def test_first_person_confession_is_blocked() -> None:
    result = check_input(
        "Ayer robé un coche, ¿qué pena me puede caer?",
        config(),
        backend=QuestionBackend(95, 5),
    )

    assert result.allowed is False
    assert result.blocked_by == ("no_self_incrimination",)
    assert result.certainty == 95.0
    assert result.support == 1
    assert "admisión" in result.reason
    assert "derechos como víctima" in result.reason


def test_both_checks_can_block_together() -> None:
    result = check_input(
        "Explícame cómo evitar que Hacienda me detecte una factura falsa",
        config(),
        backend=QuestionBackend(5, 5),
    )

    assert result.allowed is False
    assert result.blocked_by == ("legal", "no_self_incrimination")
    assert result.certainty == 95.0
    assert result.support == 2


def test_inconclusive_check_blocks_conservatively() -> None:
    result = check_input(
        "¿Me recomiendas un abogado?",
        config(threshold=70),
        backend=QuestionBackend(60, 95),
    )

    assert result.allowed is False
    assert result.blocked_by == ("legal",)
    assert result.certainty == 0.0
    assert result.support == 0
    assert "consenso" in result.reason


def test_failed_judge_blocks_that_check() -> None:
    result = check_input(
        "¿Cuál es el plazo máximo de detención preventiva?",
        config(),
        backend=QuestionBackend(RuntimeError("network down"), 95),
    )

    assert result.allowed is False
    assert result.blocked_by == ("legal",)
    assert result.checks["legal"].failed == 1
    assert "consenso" in result.reason


def test_multiple_judges_per_check() -> None:
    backend = QuestionBackend(95, 95)

    result = check_input(
        "¿Qué pena tiene robar?",
        config(judges=2),
        backend=backend,
    )

    assert result.allowed is True
    assert result.support == 4
    assert result.checks["legal"].definitive == 2
    assert result.checks["no_self_incrimination"].definitive == 2
    assert len(backend.requests) == 4


def test_empty_query_is_blocked_without_calling_the_model() -> None:
    backend = QuestionBackend(95, 95)

    result = check_input("   ", config(), backend=backend)

    assert result.allowed is False
    assert result.blocked_by == ("empty",)
    assert result.checks == {}
    assert backend.requests == []
    assert "vacía" in result.reason


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="GUARDRAIL_JUDGES"):
        InputGuardConfig(judges=0)
    with pytest.raises(ValueError, match="threshold"):
        InputGuardConfig(threshold=0)
    with pytest.raises(ValueError, match="GUARDRAIL_MODEL"):
        InputGuardConfig(model="   ")
    with pytest.raises(ValueError, match="GUARDRAIL_HISTORY_TURNS"):
        InputGuardConfig(history_turns=-1)
    with pytest.raises(TypeError, match="GUARDRAIL_HISTORY_TURNS"):
        InputGuardConfig(history_turns=True)  # type: ignore[arg-type]
    assert InputGuardConfig(history_turns=0).history_turns == 0


def test_enabled_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_INPUT_GUARDRAIL", "0")
    assert enabled() is False
    monkeypatch.setenv("AGENT_INPUT_GUARDRAIL", "yes")
    assert enabled() is True

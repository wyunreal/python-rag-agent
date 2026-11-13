from __future__ import annotations

import threading

import pytest

from decision import DecisionConfig, DecisionRequest, decide


class FakeBackend:
    """Return one configured value per request name."""

    def __init__(self, values: dict[str, float | Exception]) -> None:
        self.values = values
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def judge(self, request: DecisionRequest) -> float:
        with self._lock:
            self.calls.append(request.name)
        value = self.values[request.name]
        if isinstance(value, Exception):
            raise value
        return value


def request(name: str) -> DecisionRequest:
    return DecisionRequest(
        model="clef:27b",
        context="consulta de prueba",
        question="¿La consulta cumple la condición?",
        name=name,
    )


def test_majority_and_consolidated_certainty() -> None:
    requests = [request("a"), request("b"), request("c")]
    backend = FakeBackend({"a": 90, "b": 80, "c": 10})

    result = decide(requests, DecisionConfig(threshold=60), backend=backend)

    assert result.decision is True
    assert result.support == 2
    assert result.certainty == 60.0
    assert result.definitive == 3
    assert result.valid == 3
    assert result.failed == 0
    assert result.abstained == 0
    assert [judge.request.name for judge in result.judges] == ["a", "b", "c"]
    assert [judge.vote for judge in result.judges] == [True, True, False]


def test_below_threshold_judges_abstain() -> None:
    requests = [request("a"), request("b")]
    backend = FakeBackend({"a": 60, "b": 55})

    result = decide(requests, DecisionConfig(threshold=70), backend=backend)

    assert result.decision is False
    assert result.support == 0
    assert result.certainty == 0.0
    assert result.definitive == 0
    assert result.valid == 2
    assert result.abstained == 2


def test_fifty_fifty_has_no_vote() -> None:
    result = decide(
        [request("a")],
        DecisionConfig(threshold=50),
        backend=FakeBackend({"a": 50}),
    )

    assert result.decision is False
    assert result.support == 0
    assert result.certainty == 0.0
    assert result.definitive == 0
    assert result.judges[0].vote is None


def test_tie_blocks_and_has_fifty_percent_certainty() -> None:
    backend = FakeBackend({"a": 90, "b": 10})

    result = decide(
        [request("a"), request("b")],
        DecisionConfig(threshold=50),
        backend=backend,
    )

    assert result.decision is False
    assert result.support == 1
    assert result.certainty == 50.0
    assert result.definitive == 2


def test_one_failure_is_excluded_from_the_vote() -> None:
    backend = FakeBackend({"a": RuntimeError("boom"), "b": 90})

    result = decide(
        [request("a"), request("b")],
        DecisionConfig(threshold=60),
        backend=backend,
    )

    assert result.decision is True
    assert result.support == 1
    assert result.certainty == 90.0
    assert result.valid == 1
    assert result.failed == 1
    assert result.judges[0].error is not None
    assert "boom" in result.judges[0].error


def test_all_failures_block() -> None:
    backend = FakeBackend(
        {"a": RuntimeError("boom"), "b": RuntimeError("boom")},
    )

    result = decide(
        [request("a"), request("b")],
        DecisionConfig(threshold=60),
        backend=backend,
    )

    assert result.decision is False
    assert result.support == 0
    assert result.certainty == 0.0
    assert result.valid == 0
    assert result.failed == 2


def test_invalid_probability_is_recorded_as_an_error() -> None:
    result = decide(
        [request("a")],
        DecisionConfig(threshold=60),
        backend=FakeBackend({"a": 150}),
    )

    assert result.decision is False
    assert result.valid == 0
    assert result.judges[0].error is not None
    assert "between 0 and 100" in result.judges[0].error


def test_empty_request_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        decide([], DecisionConfig(threshold=70))


@pytest.mark.parametrize("threshold", [0, -1, 101])
def test_invalid_thresholds_are_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        DecisionConfig(threshold=threshold)


def test_threshold_must_be_a_number() -> None:
    with pytest.raises(TypeError, match="threshold"):
        DecisionConfig(threshold=True)  # type: ignore[arg-type]


def test_invalid_request_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="model"):
        DecisionRequest(model="", context="x", question="q")
    with pytest.raises(ValueError, match="question"):
        DecisionRequest(model="clef:27b", context="x", question="")
    with pytest.raises(ValueError, match="context"):
        DecisionRequest(model="clef:27b", context="  ", question="q")
    with pytest.raises(ValueError, match="context"):
        DecisionRequest(model="clef:27b", context={}, question="q")
    with pytest.raises(ValueError, match="name"):
        DecisionRequest(model="clef:27b", context="x", question="q", name="bad name")

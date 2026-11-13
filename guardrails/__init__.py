"""Input guardrail for the legal RAG agent.

The guardrail verifies that a query is a legal consultation and that it does not
contain a first-person admission of a potentially criminal fact. It is backed by
the generic :mod:`decision` engine.
"""

from guardrails.config import InputGuardConfig, enabled
from guardrails.input_guard import GuardrailResult, check_input

__all__ = [
    "GuardrailResult",
    "InputGuardConfig",
    "check_input",
    "enabled",
]

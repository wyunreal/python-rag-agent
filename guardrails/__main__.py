"""Run the input guardrail on one query.

python -m guardrails "¿Cuál es el plazo máximo de detención preventiva?"
python -m guardrails --json "Ayer robé un coche, ¿qué pena me puede caer?"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from typing import Any

from guardrails import InputGuardConfig, check_input
from guardrails.input_guard import decision_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="guardrails",
        description="Comprueba si una consulta pasa el guardrail de entrada.",
    )
    parser.add_argument("query", help="consulta a clasificar")
    parser.add_argument(
        "--threshold",
        type=float,
        help="umbral de certeza en porcentaje, mayor que 0 y hasta 100",
    )
    parser.add_argument("--model", help="modelo de decisión, por defecto clef:27b")
    parser.add_argument(
        "--judges",
        type=int,
        help="peticiones por cada uno de los dos checks",
    )
    parser.add_argument("--json", action="store_true", help="imprime la salida completa en JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = InputGuardConfig.from_env()
    overrides: dict[str, Any] = {}
    if args.threshold is not None:
        overrides["threshold"] = args.threshold
    if args.model is not None:
        overrides["model"] = args.model
    if args.judges is not None:
        overrides["judges"] = args.judges
    if overrides:
        config = replace(config, **overrides)

    try:
        result = check_input(args.query, config)
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "allowed": result.allowed,
            "certainty": result.certainty,
            "support": result.support,
            "blocked_by": list(result.blocked_by),
            "reason": result.reason,
            "checks": {name: decision_payload(check) for name, check in result.checks.items()},
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        status = "PERMITIDA" if result.allowed else "BLOQUEADA"
        print(f"{status} · certeza consolidada {result.certainty:.2f}% · support {result.support}")
        print(result.reason)
        for name, check in result.checks.items():
            print(
                f"  - {name}: decisión={check.decision} "
                f"certeza={check.certainty:.2f}% "
                f"support={check.support}/{check.definitive} definitivos"
            )
    return 0 if result.allowed else 2


if __name__ == "__main__":
    sys.exit(main())

"""Command line interface for the RAG agent.

    python -m agent index                # build/refresh the embedding index
    python -m agent ask "¿...?"          # one question, streaming answer
    python -m agent chat                 # multi-turn conversation
    python -m agent search "..."         # inspect retrieval without the LLM
    python -m agent info                 # configuration and index status

``python -m agent "¿...?"`` is shorthand for ``ask``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from time import perf_counter
from typing import Any
from uuid import uuid4

from openai import OpenAIError

from agent.config import Settings
from agent.index import (
    VectorIndex,
    build_index,
    check_embed_server,
    corpus_fingerprint,
    embed_query,
    ensure_index,
    format_hits,
)
from agent.tracing import (
    flush,
    last_trace_id,
    trace_config,
    trace_root,
    trace_url,
    tracing_enabled,
)

COMMANDS = {"index", "ask", "chat", "search", "info"}


def _text(message: Any) -> str:
    text = getattr(message, "text", "")
    if isinstance(text, str):
        return str(text)
    if callable(text):
        return str(text())

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return str(content)


def _stream(
    agent: Any,
    messages: list,
    config: dict | None = None,
    timings: bool = False,
) -> dict:
    """Stream tokens to stdout and return the final graph state."""
    from langchain_core.messages import AIMessageChunk, ToolMessage

    answer = ""
    final: dict | None = None
    started = perf_counter()
    first_llm: float | None = None
    first_tool: float | None = None
    first_text: float | None = None

    for mode, payload in agent.stream(
        {"messages": messages},
        config=config or {},
        stream_mode=["messages", "values"],
    ):
        elapsed = perf_counter() - started
        if mode == "messages":
            chunk, _meta = payload
            if isinstance(chunk, AIMessageChunk):
                if first_llm is None:
                    first_llm = elapsed
                piece = _text(chunk)
                if piece:
                    if first_text is None:
                        first_text = elapsed
                    print(piece, end="", flush=True)
                    answer += piece
            elif isinstance(chunk, ToolMessage) and first_tool is None:
                first_tool = elapsed
        elif mode == "values":
            final = payload

    if answer:
        print()
    elif final and final.get("messages"):
        answer = _text(final["messages"][-1])
        print(answer)

    if final is None:
        final = {"messages": [*messages, {"role": "assistant", "content": answer}]}

    if timings:
        parts = []
        if first_llm is not None:
            parts.append(f"first_llm={first_llm:.2f}s")
        if first_tool is not None:
            parts.append(f"tool_done={first_tool:.2f}s")
        if first_text is not None:
            parts.append(f"first_answer={first_text:.2f}s")
        parts.append(f"total={perf_counter() - started:.2f}s")
        print(f"[timings] {' '.join(parts)}")
    return final


def _invoke(
    agent: Any,
    messages: list,
    config: dict | None = None,
    timings: bool = False,
) -> dict:
    started = perf_counter()
    result = agent.invoke({"messages": messages}, config=config or {})
    elapsed = perf_counter() - started
    print(_text(result["messages"][-1]))
    if timings:
        print(f"[timings] total={elapsed:.2f}s")
    return result


def _print_trace(config: dict, trace_id: str | None = None) -> None:
    trace_id = trace_id or last_trace_id(config)
    if not trace_id:
        return
    print(f"Langfuse: {trace_url(trace_id) or trace_id}")


def _guardrail_history(messages: list, turns: int) -> list[dict[str, str]]:
    """Keep the last N user/assistant turns for the guardrail context."""
    if turns <= 0:
        return []
    filtered: list[dict[str, str]] = []
    for message in messages:
        role = {"human": "user", "ai": "assistant"}.get(getattr(message, "type", ""))
        if role is None:
            continue
        content = _text(message).strip()
        if content:
            filtered.append({"role": role, "content": content})

    user_indices = [index for index, message in enumerate(filtered) if message["role"] == "user"]
    if len(user_indices) <= turns:
        return filtered
    return filtered[user_indices[-turns] :]


def _guardrail(query: str, history: list | None = None) -> Any | None:
    """Run the input guardrail when enabled; return ``None`` if it is off."""
    from guardrails import InputGuardConfig, check_input, enabled

    if not enabled():
        return None
    config = InputGuardConfig.from_env()
    turns = _guardrail_history(history, config.history_turns) if history else None
    return check_input(query, config, history=turns)


def _cmd_index(args: argparse.Namespace, settings: Settings) -> int:
    if not args.rebuild:
        try:
            index = VectorIndex.load(settings.index_dir)
        except FileNotFoundError:
            index = None
        if index is not None and index.meta.get("fingerprint") == corpus_fingerprint(settings):
            print(
                f"Índice al día: {index.meta['count']} fragmentos, "
                f"{index.meta['dim']} dimensiones, {index.meta['embedding_model']}."
            )
            print(f"Construido: {index.meta.get('built_at', '?')} · {settings.index_dir}")
            return 0

    index = build_index(settings, show_progress_bar=not args.no_progress)
    print(
        f"Índice construido: {index.meta['count']} fragmentos, "
        f"{index.meta['dim']} dimensiones, {index.meta['embedding_model']}."
    )
    print(f"Guardado en {settings.index_dir}")
    return 0


def _cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    check_embed_server()
    index = ensure_index(settings)
    vector = embed_query(settings, args.query)
    hits = index.search(vector, k=args.k, norm_id=args.norm)
    print(format_hits(hits))
    return 0 if hits else 1


def _cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    session_id = f"ask-{uuid4().hex[:8]}"
    try:
        with trace_root(
            "ask",
            session_id=session_id,
            tags=["cli", settings.model],
            input_data={"question": args.question},
        ) as trace_context:
            verdict = _guardrail(args.question)
            if verdict is not None and not verdict.allowed:
                print(verdict.reason, file=sys.stderr)
                _print_trace({}, trace_context["trace_id"] if trace_context else None)
                return 2

            from agent.agent import build_agent

            check_embed_server()
            agent = build_agent(settings)
            messages = [{"role": "user", "content": args.question}]
            config = trace_config(
                session_id=session_id,
                trace_name="ask",
                tags=["cli", settings.model],
                trace_context=trace_context,
            )
            if args.no_stream:
                _invoke(agent, messages, config, timings=args.timings)
            else:
                _stream(agent, messages, config, timings=args.timings)
            _print_trace(config, trace_context["trace_id"] if trace_context else None)
    finally:
        flush()
    return 0


def _cmd_chat(args: argparse.Namespace, settings: Settings) -> int:
    from agent.agent import build_agent

    check_embed_server()
    agent = build_agent(settings)
    history: list = []
    session_id = f"chat-{uuid4().hex[:8]}"
    try:
        with trace_root(
            "chat",
            session_id=session_id,
            tags=["cli", settings.model],
            input_data={"session_id": session_id},
        ) as trace_context:
            root_trace_id = trace_context["trace_id"] if trace_context else None
            config = trace_config(
                session_id=session_id,
                trace_name="chat",
                tags=["cli", settings.model],
                trace_context=trace_context,
            )
            if tracing_enabled():
                base_url = os.getenv("LANGFUSE_BASE_URL", "")
                suffix = f" · {base_url}" if base_url else ""
                print(f"Langfuse: sesión {session_id}{suffix}")
            print("Agente jurídico (escribe /salir para terminar).")
            while True:
                try:
                    question = input("\nTú: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
                if not question:
                    continue
                if question.casefold() in {"/salir", "/exit", "/q"}:
                    return 0

                verdict = _guardrail(question, history)
                if verdict is not None and not verdict.allowed:
                    print(f"Agente: {verdict.reason}")
                    continue

                messages = [*history, {"role": "user", "content": question}]
                print("Agente: ", end="", flush=True)
                final = _stream(agent, messages, config, timings=args.timings)
                history = final["messages"]
                _print_trace(config, root_trace_id)
    finally:
        flush()


def _cmd_info(settings: Settings) -> int:
    print("Modelo (LLM):", settings.model)
    print("API:", settings.api_mode)
    print("Endpoint:", settings.base_url)
    print("API key:", "configurada" if settings.api_key else "NO configurada")
    print("Embeddings:", settings.embedding_model)
    print("Embeddings device (local):", settings.device)
    print("Embedding server:", os.getenv("AGENT_EMBED_SERVER") or "local")
    from guardrails import InputGuardConfig
    from guardrails import enabled as guardrail_enabled

    if guardrail_enabled():
        guardrail = InputGuardConfig.from_env()
        print(
            "Guardrail de entrada:",
            f"{guardrail.model}, umbral {guardrail.threshold:g}%, "
            f"{guardrail.judges} petición(es) por check, "
            f"{guardrail.history_turns} turno(s) de contexto",
        )
    else:
        print("Guardrail de entrada: desactivado")
    if tracing_enabled():
        print("Tracing:", f"langfuse {os.getenv('LANGFUSE_BASE_URL', '(cloud)')}")
    else:
        print("Tracing: disabled")
    print("Corpus:", settings.corpus_dir)
    print("Índice:", settings.index_dir)

    documents = sorted(settings.corpus_dir.glob("*.md"))
    print("Documentos:", len(documents))
    for document in documents:
        print("  -", document.name)

    try:
        index = VectorIndex.load(settings.index_dir)
    except FileNotFoundError:
        print("Estado del índice: no existe (ejecuta: python -m agent index)")
        return 0

    stale = index.meta.get("fingerprint") != corpus_fingerprint(settings)
    print(f"Estado del índice: {index.meta['count']} fragmentos, {index.meta['dim']} dimensiones")
    print("Construido:", index.meta.get("built_at", "?"))
    print("Actualizado:", "no, hay que reconstruirlo" if stale else "sí")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Agente RAG sobre legislación española consolidada (BOE).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="construye o actualiza el índice de embeddings")
    index.add_argument(
        "--rebuild", action="store_true", help="reconstruye aunque el índice esté al día"
    )
    index.add_argument("--no-progress", action="store_true", help="oculta la barra de progreso")
    index.set_defaults(handler=_cmd_index)

    search = sub.add_parser("search", help="muestra los fragmentos recuperados, sin LLM")
    search.add_argument("query", help="consulta en lenguaje natural")
    search.add_argument("-k", type=int, default=None, help="número de fragmentos")
    search.add_argument("--norm", help="filtra por norm_id, p. ej. BOE-A-1978-31229")
    search.set_defaults(handler=_cmd_search)

    ask = sub.add_parser("ask", help="responde una pregunta")
    ask.add_argument("question", help="pregunta en lenguaje natural")
    ask.add_argument("--no-stream", action="store_true", help="espera la respuesta completa")
    ask.add_argument("--timings", action="store_true", help="muestra tiempos por fase")
    ask.set_defaults(handler=_cmd_ask)

    chat = sub.add_parser("chat", help="conversación multi-turno")
    chat.add_argument("--timings", action="store_true", help="muestra tiempos por fase")
    chat.set_defaults(handler=_cmd_chat)

    info = sub.add_parser("info", help="configuración y estado del índice")
    info.set_defaults(handler=lambda _args, settings: _cmd_info(settings))

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.getenv("AGENT_LOG_LEVEL", "INFO").upper(),
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    for noisy in (
        "httpx",
        "httpx2",
        "httpcore",
        "huggingface_hub",
        "langchain_openai",
        "sentence_transformers",
        "transformers",
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "ask")

    parser = _parser()
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    settings = Settings.from_env()

    if getattr(args, "k", None) is None:
        args.k = settings.top_k

    try:
        return int(args.handler(args, settings))
    except (RuntimeError, FileNotFoundError, ValueError, OpenAIError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

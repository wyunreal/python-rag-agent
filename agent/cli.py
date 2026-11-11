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
from typing import Any

from openai import OpenAIError

from agent.config import Settings
from agent.index import VectorIndex, build_index, corpus_fingerprint, format_hits, get_embedder

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


def _stream(agent: Any, messages: list) -> dict:
    """Stream tokens to stdout and return the final graph state."""
    from langchain_core.messages import AIMessageChunk

    answer = ""
    final: dict | None = None
    for mode, payload in agent.stream(
        {"messages": messages},
        stream_mode=["messages", "values"],
    ):
        if mode == "messages":
            chunk, _meta = payload
            if isinstance(chunk, AIMessageChunk):
                piece = _text(chunk)
                if piece:
                    print(piece, end="", flush=True)
                    answer += piece
        elif mode == "values":
            final = payload

    if answer:
        print()
    elif final and final.get("messages"):
        answer = _text(final["messages"][-1])
        print(answer)

    if final is None:
        final = {"messages": [*messages, {"role": "assistant", "content": answer}]}
    return final


def _invoke(agent: Any, messages: list) -> dict:
    result = agent.invoke({"messages": messages})
    print(_text(result["messages"][-1]))
    return result


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
    from agent.index import ensure_index

    index = ensure_index(settings)
    vector = get_embedder(settings).encode([args.query])[0]
    hits = index.search(vector, k=args.k, norm_id=args.norm)
    print(format_hits(hits))
    return 0 if hits else 1


def _cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    from agent.agent import build_agent

    agent = build_agent(settings)
    messages = [{"role": "user", "content": args.question}]
    if args.no_stream:
        _invoke(agent, messages)
    else:
        _stream(agent, messages)
    return 0


def _cmd_chat(args: argparse.Namespace, settings: Settings) -> int:
    from agent.agent import build_agent

    agent = build_agent(settings)
    history: list = []
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

        messages = [*history, {"role": "user", "content": question}]
        print("Agente: ", end="", flush=True)
        final = _stream(agent, messages)
        history = final["messages"]


def _cmd_info(settings: Settings) -> int:
    print("Modelo (LLM):", settings.model)
    print("API:", settings.api_mode)
    print("Endpoint:", settings.base_url)
    print("API key:", "configurada" if settings.api_key else "NO configurada")
    print("Embeddings:", settings.embedding_model)
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
    ask.set_defaults(handler=_cmd_ask)

    chat = sub.add_parser("chat", help="conversación multi-turno")
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

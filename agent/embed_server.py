"""Persistent HTTP embedding server.

The model is loaded once at startup; the CLI then asks this process for query
vectors instead of importing torch and loading bge-m3 on every ``ask``.
"""

from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent.config import Settings
from agent.index import get_embedder

logger = logging.getLogger(__name__)
SETTINGS = Settings.from_env()
_EMBEDDER = None


def embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        logger.info(
            "Loading embedding model %s on %s…",
            SETTINGS.embedding_model,
            SETTINGS.device,
        )
        _EMBEDDER = get_embedder(SETTINGS)
    return _EMBEDDER


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if self.path != "/embed":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            texts = data.get("texts")
            if (
                not isinstance(texts, list)
                or not texts
                or not all(isinstance(text, str) for text in texts)
            ):
                self._json(400, {"error": "texts must be a non-empty list of strings"})
                return
            vectors = embedder().encode(texts, show_progress_bar=False)
            self._json(200, {"vectors": vectors.tolist()})
        except Exception as exc:  # noqa: BLE001 - return a JSON error to the client
            logger.exception("Embedding request failed")
            self._json(500, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        logger.info("%s - %s", self.address_string(), format % args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Persistent embedding server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(message)s",
    )
    embedder()  # fail before accepting requests if the model cannot load
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    logger.info("Embedding server listening on %s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

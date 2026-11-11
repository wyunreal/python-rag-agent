"""Embedding index over the corpus: build, persist, load and query.

The index is a plain ``embeddings.npy`` matrix plus ``chunks.jsonl`` and a
small ``meta.json``. No vector database: with a handful of norms this is faster,
easier to inspect and trivial to rebuild when the corpus changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

from agent.config import Settings
from agent.corpus import CHUNKER_VERSION, MAX_CHARS, OVERLAP, Chunk, load_corpus

logger = logging.getLogger(__name__)

CHUNKS_FILE = "chunks.jsonl"
EMBEDDINGS_FILE = "embeddings.npy"
META_FILE = "meta.json"


@dataclass(frozen=True, slots=True)
class SearchHit:
    score: float
    chunk: Chunk


def corpus_fingerprint(settings: Settings) -> str:
    """Hash of the corpus files, chunking parameters and embedding model."""
    digest = hashlib.sha256()
    digest.update(settings.embedding_model.encode())
    digest.update(f"{CHUNKER_VERSION}:{MAX_CHARS}:{OVERLAP}".encode())
    pattern = ("*.md", "*.index.jsonl")
    paths = sorted(path for glob in pattern for path in settings.corpus_dir.glob(glob))
    if not paths:
        raise FileNotFoundError(
            f"No hay documentos en {settings.corpus_dir}. Ejecuta: python -m boe_ingest"
        )
    for path in paths:
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


class Embedder:
    """Thin wrapper around sentence-transformers.

    Imported lazily so commands that do not touch the index (for example
    ``python -m agent info``) start without loading torch.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        from transformers.utils import logging as transformers_logging

        # Model loading is noisy and not useful to the CLI user.
        for noisy in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers"):
            logging.getLogger(noisy).setLevel(logging.ERROR)
        transformers_logging.disable_progress_bar()

        self.model_name = model_name
        self._device = device or "cpu"
        self._fallback_warned = False
        self._lock = threading.Lock()
        self._model = self._load(self._device)

    def _warn_fallback(self) -> None:
        if not self._fallback_warned:
            logger.warning("CUDA out of memory; using CPU for embeddings.")
            self._fallback_warned = True

    def _load(self, device: str):
        from sentence_transformers import SentenceTransformer

        try:
            return SentenceTransformer(self.model_name, device=device, local_files_only=True)
        except RuntimeError as exc:
            if device != "cpu" and "out of memory" in str(exc).lower():
                self._warn_fallback()
                self._device = "cpu"
                return SentenceTransformer(self.model_name, device="cpu", local_files_only=True)
            raise
        except (OSError, ValueError):
            logger.info("Embedding model not cached; downloading %s.", self.model_name)
            return SentenceTransformer(self.model_name, device=device)

    def _encode(
        self,
        texts: list[str],
        batch_size: int,
        show_progress_bar: bool,
    ) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        with self._lock:
            try:
                vectors = self._encode(texts, batch_size, show_progress_bar)
            except RuntimeError as exc:
                if self._device != "cpu" and "out of memory" in str(exc).lower():
                    self._warn_fallback()
                    self._device = "cpu"
                    self._model = self._load("cpu")
                    vectors = self._encode(texts, batch_size, show_progress_bar)
                else:
                    raise
        return np.asarray(vectors, dtype=np.float32)


_EMBEDDER_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _shared_embedder(model_name: str, device: str | None) -> Embedder:
    return Embedder(model_name, device)


def get_embedder(settings: Settings) -> Embedder:
    # ToolNode may run several search_corpus calls in parallel.
    with _EMBEDDER_LOCK:
        return _shared_embedder(settings.embedding_model, settings.device)


@lru_cache(maxsize=1024)
def _cached_query_vector(model_name: str, device: str | None, text: str) -> np.ndarray:
    return _shared_embedder(model_name, device).encode([text])[0]


@lru_cache(maxsize=1024)
def _cached_server_vector(base_url: str, model_name: str, text: str) -> np.ndarray:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/embed",
        data=json.dumps({"texts": [text]}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    vectors = np.asarray(payload["vectors"], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != 1:
        raise ValueError("unexpected response from embedding server")
    return vectors[0]


_QUERY_LOCK = threading.Lock()


def check_embed_server() -> None:
    """Fail fast when a configured embedding server is not reachable."""
    server = os.getenv("AGENT_EMBED_SERVER", "").strip()
    if not server:
        return

    health_url = f"{server.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=2) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
    except Exception as exc:  # noqa: BLE001 - wrap any connection error
        raise RuntimeError(
            f"El servidor de embeddings {server} no responde. "
            "Arráncalo con `make embed-up` o quita AGENT_EMBED_SERVER para usar "
            "el modelo local."
        ) from exc


def embed_query(settings: Settings, text: str) -> np.ndarray:
    """Embed a query via the persistent server, or locally when no server is set."""
    server = os.getenv("AGENT_EMBED_SERVER", "").strip()
    if server:
        with _QUERY_LOCK:
            try:
                return _cached_server_vector(server, settings.embedding_model, text)
            except Exception as exc:  # noqa: BLE001 - no silent fallback when configured
                raise RuntimeError(
                    f"El servidor de embeddings {server} no responde. "
                    "Arráncalo con `make embed-up` o quita AGENT_EMBED_SERVER para "
                    "usar el modelo local."
                ) from exc
    with _QUERY_LOCK:
        return _cached_query_vector(settings.embedding_model, settings.device, text)


class VectorIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, meta: dict) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks.jsonl y embeddings.npy tienen tamaños distintos")
        self.chunks = chunks
        self.embeddings = embeddings
        self.meta = meta

    @classmethod
    def load(cls, index_dir: Path) -> VectorIndex:
        chunks_path = index_dir / CHUNKS_FILE
        if not chunks_path.exists():
            raise FileNotFoundError(f"No hay índice en {index_dir}. Ejecuta: python -m agent index")
        with chunks_path.open(encoding="utf-8") as handle:
            chunks = [Chunk.from_dict(json.loads(line)) for line in handle if line.strip()]
        embeddings = np.load(index_dir / EMBEDDINGS_FILE)
        meta = json.loads((index_dir / META_FILE).read_text(encoding="utf-8"))
        return cls(chunks, embeddings, meta)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        np.save(index_dir / EMBEDDINGS_FILE, self.embeddings)
        (index_dir / META_FILE).write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        norm_id: str | None = None,
    ) -> list[SearchHit]:
        scores = self.embeddings @ np.asarray(query_vector, dtype=np.float32)
        if norm_id:
            allowed = np.array([chunk.norm_id == norm_id for chunk in self.chunks])
            if not allowed.any():
                return []
            scores = np.where(allowed, scores, -np.inf)
            k = min(k, int(allowed.sum()))
        k = max(1, min(k, len(self.chunks)))
        best = np.argsort(-scores)[:k]
        return [SearchHit(float(scores[i]), self.chunks[i]) for i in best]


def build_index(settings: Settings, show_progress_bar: bool = True) -> VectorIndex:
    """Embed the whole corpus and persist the index."""
    chunks = load_corpus(settings.corpus_dir)
    logger.info("Indexando %d fragmentos con %s…", len(chunks), settings.embedding_model)
    embedder = get_embedder(settings)
    vectors = embedder.encode(
        [chunk.embed_text() for chunk in chunks],
        batch_size=settings.batch_size,
        show_progress_bar=show_progress_bar,
    )
    meta = {
        "embedding_model": settings.embedding_model,
        "dim": int(vectors.shape[1]),
        "count": len(chunks),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "fingerprint": corpus_fingerprint(settings),
        "chunking": {
            "version": CHUNKER_VERSION,
            "max_chars": MAX_CHARS,
            "overlap": OVERLAP,
        },
    }
    index = VectorIndex(chunks, vectors, meta)
    index.save(settings.index_dir)
    logger.info("Índice guardado en %s", settings.index_dir)
    return index


_INDEX_CACHE: dict[Path, tuple[str, VectorIndex]] = {}
_INDEX_LOCK = threading.Lock()


def ensure_index(settings: Settings, force: bool = False) -> VectorIndex:
    """Return the index, rebuilding it when missing or stale.

    The loaded index is cached in memory per index directory. A cheap
    fingerprint check still runs on every call, so a corpus change triggers a
    rebuild while repeated tool calls avoid reading the files again.
    """
    fingerprint = corpus_fingerprint(settings)
    with _INDEX_LOCK:
        cached = _INDEX_CACHE.get(settings.index_dir)
        if not force and cached is not None and cached[0] == fingerprint:
            return cached[1]

        if force:
            index = build_index(settings)
        else:
            try:
                index = VectorIndex.load(settings.index_dir)
            except FileNotFoundError:
                logger.info("No existe índice; construyéndolo por primera vez.")
                index = build_index(settings)
            else:
                if index.meta.get("fingerprint") != fingerprint:
                    logger.info("El corpus o el modelo han cambiado; reconstruyendo el índice.")
                    index = build_index(settings)

        _INDEX_CACHE[settings.index_dir] = (fingerprint, index)
        return index


def format_hits(hits: list[SearchHit]) -> str:
    """Render results for the LLM, with enough metadata to cite them."""
    if not hits:
        return "No se han encontrado fragmentos para esa consulta."

    lines: list[str] = []
    for position, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        context = f"{chunk.section} · " if chunk.section else ""
        validity = f" · vigencia {chunk.in_force}" if chunk.in_force else ""
        lines.append(
            f"[{position}] {chunk.label()}\n"
            f"{context}{chunk.norm_id}{validity}\n"
            f"{chunk.url}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(lines)

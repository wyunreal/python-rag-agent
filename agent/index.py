"""Embedding index over the corpus: build, persist, load and query.

The index is a plain ``embeddings.npy`` matrix plus ``chunks.jsonl`` and a
small ``meta.json``. No vector database: with a handful of norms this is faster,
easier to inspect and trivial to rebuild when the corpus changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
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
            return SentenceTransformer(self.model_name, device=device)
        except RuntimeError as exc:
            if device != "cpu" and "out of memory" in str(exc).lower():
                self._warn_fallback()
                self._device = "cpu"
                return SentenceTransformer(self.model_name, device="cpu")
            raise

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


def ensure_index(settings: Settings, force: bool = False) -> VectorIndex:
    """Load the index, rebuilding it when missing or stale."""
    if force:
        return build_index(settings)
    try:
        index = VectorIndex.load(settings.index_dir)
    except FileNotFoundError:
        logger.info("No existe índice; construyéndolo por primera vez.")
        return build_index(settings)
    if index.meta.get("fingerprint") != corpus_fingerprint(settings):
        logger.info("El corpus o el modelo han cambiado; reconstruyendo el índice.")
        return build_index(settings)
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

"""Runtime configuration for the RAG agent.

Everything comes from environment variables (or a ``.env`` file) so the code
never contains credentials. The corpus and index live under ``data/`` by
default, next to the download produced by ``boe_ingest``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Keep Hugging Face model loading quiet; the CLI has its own logging.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = ROOT / "data" / "corpus"
DEFAULT_INDEX_DIR = ROOT / "data" / "index"

DEFAULT_BASE_URL = "https://ollama.com/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash:1017"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_API_MODE = "responses"
API_MODES = {"responses", "chat_completions"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable settings snapshot.

    Creating it never fails because the API key is missing: indexing and the
    ``search`` command work offline. Only ``require_api_key`` enforces it.
    """

    api_key: str
    base_url: str
    model: str
    api_mode: str
    embedding_model: str
    corpus_dir: Path
    index_dir: Path
    top_k: int
    batch_size: int
    device: str | None

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(ROOT / ".env")
        api_mode = os.getenv("AGENT_API_MODE", DEFAULT_API_MODE).strip().lower()
        if api_mode not in API_MODES:
            raise ValueError(f"AGENT_API_MODE debe ser uno de {sorted(API_MODES)}, no {api_mode!r}")
        return cls(
            api_key=os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_CLOUD_API_KEY", ""),
            base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
            model=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
            api_mode=api_mode,
            embedding_model=os.getenv("AGENT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            corpus_dir=_path("AGENT_CORPUS_DIR", DEFAULT_CORPUS_DIR),
            index_dir=_path("AGENT_INDEX_DIR", DEFAULT_INDEX_DIR),
            top_k=_int("AGENT_TOP_K", 5),
            batch_size=_int("AGENT_BATCH_SIZE", 32),
            device=os.getenv("AGENT_DEVICE") or None,
        )

    def require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "Falta OLLAMA_API_KEY. Crea una clave en https://ollama.com/settings/keys "
                "y añádela a tu .env (ver .env.example)."
            )
        return self.api_key

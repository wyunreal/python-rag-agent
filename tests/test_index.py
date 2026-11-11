from __future__ import annotations

import numpy as np

from agent.corpus import Chunk
from agent.index import VectorIndex


def _chunk(chunk_id: str, norm_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        norm_id=norm_id,
        title="Norma",
        heading=f"Artículo {chunk_id}",
        section="",
        kind="precepto",
        url="https://example.test",
        text=f"Texto {chunk_id}",
    )


def test_filter_never_returns_chunks_from_other_norms() -> None:
    chunks = [_chunk("1", "A"), _chunk("2", "B"), _chunk("3", "B")]
    index = VectorIndex(chunks, np.eye(3, dtype=np.float32), {})

    hits = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=10, norm_id="A")

    assert [hit.chunk.norm_id for hit in hits] == ["A"]
    assert hits[0].chunk.id == "1"


def test_unknown_norm_returns_nothing() -> None:
    chunks = [_chunk("1", "A")]
    index = VectorIndex(chunks, np.eye(1, dtype=np.float32), {})

    assert index.search(np.array([1.0], dtype=np.float32), k=5, norm_id="ZZZ") == []

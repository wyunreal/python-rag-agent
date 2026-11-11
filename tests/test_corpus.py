from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.corpus import MAX_CHARS, load_corpus

FRONTMATTER = """\
---
id: TEST-1
title: Norma de prueba.
source_document_url: https://example.test/norma
---
> Basado en datos de la Agencia Estatal Boletín Oficial del Estado.
>
> Texto consolidado de carácter meramente informativo.

# Norma de prueba

### TÍTULO I — De prueba

#### CAPÍTULO ÚNICO

###### Artículo 1

{body}
"""


def _write_norm(tmp_path: Path, body: str) -> Path:
    (tmp_path / "TEST-1.md").write_text(FRONTMATTER.format(body=body), encoding="utf-8")
    row = {
        "block_id": "a1",
        "heading": "Artículo 1",
        "kind": "precepto",
        "in_force": "2020-01-01",
        "url": "https://example.test/norma/bloque/a1",
    }
    (tmp_path / "TEST-1.index.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return tmp_path


def test_licence_block_is_not_indexed(tmp_path: Path) -> None:
    chunks = load_corpus(_write_norm(tmp_path, "Texto del artículo uno."))
    assert len(chunks) == 1
    assert chunks[0].text == "Texto del artículo uno."
    assert "Boletín Oficial" not in chunks[0].text


def test_article_keeps_hierarchy_and_citation(tmp_path: Path) -> None:
    chunk = load_corpus(_write_norm(tmp_path, "Texto del artículo uno."))[0]
    assert chunk.heading == "Artículo 1"
    assert chunk.section == "TÍTULO I — De prueba › CAPÍTULO ÚNICO"
    assert chunk.kind == "precepto"
    assert chunk.url == "https://example.test/norma/bloque/a1"
    assert chunk.in_force == "2020-01-01"
    assert chunk.label() == "Norma de prueba — Artículo 1"


def test_long_articles_are_split_within_the_limit(tmp_path: Path) -> None:
    body = "\n\n".join(f"Párrafo {number}." for number in range(400))
    chunks = load_corpus(_write_norm(tmp_path, body))
    assert len(chunks) > 1
    assert all(len(chunk.text) <= MAX_CHARS for chunk in chunks)
    assert all(chunk.heading == "Artículo 1" for chunk in chunks)
    assert chunks[0].id.endswith(":1")


def test_missing_corpus_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="boe_ingest"):
        load_corpus(tmp_path)

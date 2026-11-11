"""Turn the rendered Markdown corpus into citation-ready chunks.

The Markdown produced by ``boe_ingest`` has a YAML frontmatter block and a
heading hierarchy (``##`` book, ``###`` title, ``####`` chapter, ``#####``
section, ``######`` article). We cut on headings, keep the enclosing hierarchy
as metadata, and only split blocks that are too long to embed as a single unit.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$")
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
SENTENCE_RE = re.compile(r"(?<=[.;:!?])\s+")
SLUG_RE = re.compile(r"\W+")

# bge-m3 accepts long inputs, but legal retrieval works better with one
# precept (or a few short ones) per vector.
MAX_CHARS = 1800
OVERLAP = 200
# Bump whenever the chunking logic changes so stale indexes are rebuilt.
CHUNKER_VERSION = 1

DEFAULT_HEADING = "Preámbulo"


@dataclass(frozen=True, slots=True)
class Chunk:
    """One embeddable, citable unit of the corpus."""

    id: str
    norm_id: str
    title: str
    heading: str
    section: str
    kind: str
    url: str
    text: str
    in_force: str = ""

    def embed_text(self) -> str:
        """Text handed to the embedding model: headings give it context."""
        parts = [self.title]
        if self.section:
            parts.append(self.section)
        if self.heading:
            parts.append(self.heading)
        return "\n".join(parts) + "\n\n" + self.text

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "norm_id": self.norm_id,
            "title": self.title,
            "heading": self.heading,
            "section": self.section,
            "kind": self.kind,
            "url": self.url,
            "text": self.text,
            "in_force": self.in_force,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Chunk:
        return cls(
            id=data["id"],
            norm_id=data["norm_id"],
            title=data["title"],
            heading=data["heading"],
            section=data["section"],
            kind=data["kind"],
            url=data["url"],
            text=data["text"],
            in_force=data.get("in_force", ""),
        )

    def label(self) -> str:
        """Human-readable citation such as ``Código Civil — Artículo 1``."""
        return f"{self.title} — {self.heading}" if self.heading else self.title


@dataclass(frozen=True, slots=True)
class _Block:
    heading: str
    section: str
    kind: str
    text: str


def _clean(heading: str) -> str:
    """Normalize headings for matching the index against the Markdown."""
    text = heading.replace("*", "").replace("_", " ").strip()
    text = re.sub(r"[.\s]+$", "", text)
    return re.sub(r"\s+", " ", text)


def _key(heading: str) -> str:
    return _clean(heading).casefold()


def _read(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(raw)
    if not match:
        raise ValueError(f"{path}: falta el frontmatter YAML")
    meta = yaml.safe_load(match.group(1)) or {}
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: frontmatter inválido")
    return meta, raw[match.end() :]


def _block_index(path: Path) -> dict[str, dict]:
    """Map ``heading`` -> index row (url, kind, in_force) for one norm."""
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = _key(str(row.get("heading", "")))
            if key and key not in rows:
                rows[key] = row
    return rows


def _tail(text: str, size: int) -> str:
    if len(text) <= size:
        return text
    tail = text[-size:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def _units(text: str) -> Iterator[str]:
    """Yield paragraphs, splitting oversized ones on sentence boundaries."""
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= MAX_CHARS:
            yield paragraph
            continue
        for sentence in SENTENCE_RE.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                yield sentence


def split_text(text: str) -> list[str]:
    """Pack text into ``MAX_CHARS`` pieces with a small overlap."""
    chunks: list[str] = []
    current = ""
    for unit in _units(text):
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= MAX_CHARS:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = f"{_tail(current, OVERLAP)}\n\n{unit}"
        else:
            current = unit
        while len(current) > MAX_CHARS:
            chunks.append(current[:MAX_CHARS])
            current = _tail(current[MAX_CHARS - OVERLAP :], MAX_CHARS)
    if current.strip():
        chunks.append(current)
    return chunks


def _iter_blocks(body: str) -> Iterator[_Block]:
    """Split the document body on headings, remembering the hierarchy."""
    lines = body.splitlines(keepends=True)
    # Drop the licence block that precedes the document title.
    for position, line in enumerate(lines):
        if line.startswith("# "):
            lines = lines[position + 1 :]
            break

    stack: dict[int, str] = {}
    heading, section, kind = DEFAULT_HEADING, "", "preambulo"
    buffer: list[str] = []

    for line in lines:
        match = HEADING_RE.match(line)
        if not match:
            if not line.startswith("# "):
                buffer.append(line)
            continue

        text = "".join(buffer).strip()
        buffer.clear()
        if text:
            yield _Block(heading, section, kind, text)

        level = len(match.group(1))
        heading = _clean(match.group(2))
        stack = {depth: name for depth, name in stack.items() if depth < level}
        section = " › ".join(stack[depth] for depth in sorted(stack))
        stack[level] = heading
        kind = "precepto" if level >= 6 else "encabezado"

    text = "".join(buffer).strip()
    if text:
        yield _Block(heading, section, kind, text)


def _iter_chunks(path: Path) -> Iterator[Chunk]:
    meta, body = _read(path)
    norm_id = str(meta.get("id") or path.stem)
    title = str(meta.get("title") or norm_id).strip().rstrip(".")
    doc_url = str(meta.get("source_document_url") or "")
    blocks = _block_index(path.with_suffix(".index.jsonl"))

    used_ids: set[str] = set()
    for block in _iter_blocks(body):
        if not block.text:
            continue

        key = _key(block.heading)
        row = blocks.get(key, {})
        block_id = str(row.get("block_id") or SLUG_RE.sub("-", key).strip("-") or "block")
        parts = split_text(block.text)

        for part_no, part in enumerate(parts, start=1):
            suffix = f":{part_no}" if len(parts) > 1 else ""
            chunk_id = f"{norm_id}:{block_id}{suffix}"
            while chunk_id in used_ids:
                chunk_id += "-x"
            used_ids.add(chunk_id)
            yield Chunk(
                id=chunk_id,
                norm_id=norm_id,
                title=title,
                heading=block.heading,
                section=block.section,
                kind=str(row.get("kind") or block.kind),
                url=str(row.get("url") or doc_url),
                text=part,
                in_force=str(row.get("in_force") or ""),
            )


def load_corpus(corpus_dir: Path) -> list[Chunk]:
    """Read every ``*.md`` norm and return all chunks in deterministic order."""
    paths = sorted(corpus_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(
            f"No hay documentos en {corpus_dir}. Ejecuta primero: python -m boe_ingest"
        )
    chunks: list[Chunk] = []
    for path in paths:
        chunks.extend(_iter_chunks(path))
    return chunks

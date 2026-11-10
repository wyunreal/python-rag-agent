"""Render BOE consolidated-legislation XML as a single Markdown document.

No chunking happens here. One norm becomes one Markdown file with the legal
hierarchy expressed as headings, so a later stage can split it on structure
instead of guessing where articles begin and end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from xml.etree import ElementTree as ET

REUSE_NOTICE = (
    "Texto consolidado de carácter meramente informativo. "
    "La versión auténtica es la publicada en el «Boletín Oficial del Estado»."
)

# Required wording for derivative reuse (BOE reuse conditions, 27 June 2024,
# condition fourth): adaptations must cite "Basado en datos de..." and link
# to boe.es, and must identify the modifications made.
ATTRIBUTION = (
    "Basado en datos de la "
    "[Agencia Estatal Boletín Oficial del Estado](https://www.boe.es)."
)

_WHITESPACE = re.compile(r"\s+")

# Legal hierarchy -> Markdown heading depth. Subsección shares Sección's level
# so that Artículo can keep the deepest level (######).
_PAIRED_HEADINGS = {"libro": 2, "titulo": 3, "capitulo": 4}
_SINGLE_HEADINGS = {"titulo": 3, "seccion": 5, "subseccion": 5, "anexo": 3}
_DISPLAY_TITLES = {"centro_negrita", "centro_redonda"}


def _inline(element: ET.Element) -> str:
    """Serialise an element's text, keeping the inline markup that matters."""
    parts = [element.text or ""]
    for child in element:
        inner = _inline(child)
        if child.tag == "strong" and inner.strip():
            parts.append(f"**{inner}**")
        elif child.tag == "em" and inner.strip():
            parts.append(f"*{inner}*")
        elif child.tag == "a" and (child.get("href") or "").strip() and inner.strip():
            parts.append(f"[{inner}]({child.get('href').strip()})")
        else:
            parts.append(inner)
        parts.append(child.tail or "")
    return "".join(parts)


def _text(element: ET.Element) -> str:
    return _WHITESPACE.sub(" ", _inline(element)).strip()


def iso_date(raw: str | None) -> str | None:
    """BOE dates are YYYYMMDD; present them as YYYY-MM-DD."""
    if not raw or len(raw) != 8 or not raw.isdigit():
        return raw
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


def iso_timestamp(raw: str | None) -> str | None:
    """BOE timestamps are YYYYMMDDThhmmssZ; present them as ISO 8601."""
    if not raw or len(raw) != 16 or raw[8] != "T" or raw[-1] != "Z":
        return raw
    date, time = raw[:8], raw[9:15]
    return f"{iso_date(date)}T{time[:2]}:{time[2:4]}:{time[4:6]}Z"


def _applicable_version(block: ET.Element, as_of: str) -> ET.Element | None:
    """The version in force on `as_of` (YYYYMMDD), or None if none is yet.

    A block can carry versions that only enter into force in the future. Taking
    the last one would render pending law as current, so future versions are
    filtered out before choosing.
    """
    versions = block.findall("version")
    if not versions:
        return None

    in_force = [
        version
        for version in versions
        if (version.get("fecha_vigencia") or "00000000") <= as_of
    ]
    if not in_force:
        return None
    return max(in_force, key=lambda version: version.get("fecha_vigencia") or "")


def _is_expired(block: ET.Element, as_of: str) -> bool:
    expiry = block.get("fecha_caducidad")
    return bool(expiry and expiry <= as_of)


def _render_paragraphs(paragraphs: list[ET.Element]) -> list[str]:
    lines: list[str] = []
    index = 0

    while index < len(paragraphs):
        paragraph = paragraphs[index]
        css = (paragraph.get("class") or "").strip()
        text = _text(paragraph)

        if css.endswith("_num"):
            base = css[: -len("_num")]
            level = _PAIRED_HEADINGS.get(base)
            if level is None:
                if text:
                    lines.append(text)
                index += 1
                continue

            title = ""
            if index + 1 < len(paragraphs):
                following = (paragraphs[index + 1].get("class") or "").strip()
                if following == f"{base}_tit":
                    title = _text(paragraphs[index + 1])
                    index += 1

            heading = f"{text} — {title}" if text and title else (text or title)
            if heading:
                lines.append(f"{'#' * level} {heading}")

        elif css == "articulo":
            if text:
                lines.append(f"###### {text}")

        elif css in _SINGLE_HEADINGS:
            if text:
                lines.append(f"{'#' * _SINGLE_HEADINGS[css]} {text}")

        elif css in _DISPLAY_TITLES:
            if text:
                lines.append(f"**{text}**")

        elif css == "cita_con_pleca":
            if text:
                lines.append(f"> {text}")

        elif css.startswith("firma_"):
            if text:
                lines.append(f"*{text}*")

        elif text:
            lines.append(text)

        index += 1

    return lines


def _yaml(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_yaml(item) for item in value) + "]"
    if isinstance(value, int):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _front_matter(meta: dict[str, object]) -> str:
    lines = ["---"]
    lines += [f"{key}: {_yaml(value)}" for key, value in meta.items()]
    lines.append("---")
    return "\n".join(lines)


@dataclass(frozen=True)
class RenderedDocument:
    """One norm rendered for a reference date.

    `pending_markdown` holds the wording that is published but not yet in force.
    Keeping it in a separate file means the main corpus stays "law in force",
    while a retrieval layer can opt in to the future text.

    `index` is one record per rendered block, so a later chunker can carry the
    provenance of each article instead of only the document's.
    """

    markdown: str
    pending_markdown: str | None
    index: list[dict[str, object]]
    stats: dict[str, object]


def _heading_of(rendered: list[str], block: ET.Element) -> str:
    """Best label for a block, for the provenance index."""
    if block.get("id") == "preambulo" or block.get("tipo") == "preambulo":
        return "Preámbulo"
    if block.get("tipo") == "firma":
        return "Firma"
    if block.get("tipo") == "nota_inicial":
        return "Nota editorial"
    for line in rendered:
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return block.get("titulo") or ""


def _pending_note(future: list[ET.Element]) -> str:
    """One-line marker for a block whose next wording is not yet in force."""
    first = future[0]
    note = (
        "> **Reforma pendiente** — nueva redacción con efectos desde "
        f"{iso_date(first.get('fecha_vigencia'))}"
    )
    published = iso_date(first.get("fecha_publicacion"))
    if published:
        note += f", publicada el {published}"
    if len(future) > 1:
        note += f"; {len(future) - 1} versión(es) posterior(es)"
    return note + ". No vigente todavía."


def _block_heading(block: ET.Element, versions: list[ET.Element]) -> str:
    """Prefer the article's own title line over BOE's short `titulo` attribute."""
    for version in versions:
        for paragraph in version.findall("p"):
            if (paragraph.get("class") or "") == "articulo":
                text = _text(paragraph)
                if text:
                    return text
        break
    return block.get("titulo") or block.get("id") or ""


def _notice(meta: dict[str, object], *, pending: bool = False) -> str:
    """Reuse notice shared by the main and the pending documents."""
    updated = str(meta.get("last_updated") or "").split("T")[0]
    document_line = f"> Documento: {meta['id']}"
    if updated:
        document_line += f" · última actualización {updated}"
    document_line += f" · versión consultada {meta['as_of']}"

    lines = [
        f"> {ATTRIBUTION}",
        ">",
        f"> {REUSE_NOTICE}",
    ]
    if meta.get("consolidation_stale"):
        lines += [
            ">",
            "> **Aviso: el BOE marca este texto consolidado como desactualizado.**",
            "> Puede no incorporar modificaciones recientes.",
        ]
    if pending:
        lines += [
            ">",
            "> Documento de reformas publicadas y **no vigentes** a "
            f"{meta['as_of']}. No debe citarse como derecho vigente.",
        ]
    lines.append(">")
    lines.append(document_line)
    if meta.get("html_url"):
        lines.append(f"> {meta['html_url']}")
    lines += [
        ">",
        "> Formato y estructura adaptados a Markdown por python-rag-agent.",
        "> Proyecto independiente del BOE; no está respaldado por él.",
    ]
    return "\n".join(lines)


def _pending_document(
    meta: dict[str, object],
    sections: list[tuple[ET.Element, list[ET.Element]]],
) -> str | None:
    if not sections:
        return None

    parts = [
        _front_matter({**meta, "pending": True}),
        _notice(meta, pending=True),
        f"# Reformas pendientes — {meta['title']}",
    ]

    for block, future in sections:
        parts.append(f"## {_block_heading(block, future)}")
        for version in future:
            parts.append(
                "**Efectos desde "
                f"{iso_date(version.get('fecha_vigencia'))}**"
                + (
                    f" (publicada el {iso_date(version.get('fecha_publicacion'))})"
                    if version.get("fecha_publicacion")
                    else ""
                )
            )
            parts.extend(_render_paragraphs(version.findall("p")))

    return "\n\n".join(parts).rstrip() + "\n"


def render_document(
    meta: dict[str, object],
    data: ET.Element,
    *,
    as_of: str,
) -> RenderedDocument:
    """Render one consolidated norm as of `as_of` (YYYYMMDD).

    `data` is the <data> element of a /texto response. The main document holds
    the law in force; wording that only enters into force later goes to a
    separate pending document. Repealed and not-yet-in-force blocks are counted
    so nothing disappears silently.
    """
    blocks = list(data.iter("bloque"))

    in_force: list[tuple[ET.Element, ET.Element, list[ET.Element]]] = []
    pending_sections: list[tuple[ET.Element, list[ET.Element]]] = []
    expired_ids: list[str] = []
    pending_ids: list[str] = []
    amended_ids: list[str] = []
    versions_seen = 0
    future_versions = 0

    for block in blocks:
        versions = block.findall("version")
        versions_seen += len(versions)

        if _is_expired(block, as_of):
            expired_ids.append(block.get("id") or "")
            continue

        future = sorted(
            (
                version
                for version in versions
                if (version.get("fecha_vigencia") or "00000000") > as_of
            ),
            key=lambda version: version.get("fecha_vigencia") or "",
        )
        future_versions += len(future)

        version = _applicable_version(block, as_of)
        if version is None:
            # Nothing in force yet: the whole block is future law.
            pending_ids.append(block.get("id") or "")
            pending_sections.append((block, future))
            continue

        if future:
            amended_ids.append(block.get("id") or "")
            pending_sections.append((block, future))

        in_force.append((block, version, future))

    body: list[str] = []
    index: list[dict[str, object]] = []
    for block, version, future in in_force:
        rendered = _render_paragraphs(version.findall("p"))
        if block.get("tipo") == "nota_inicial":
            # Editorial note (e.g. "includes the errata published in..."), not law text.
            rendered = [f"> {line}" for line in rendered]
        if future and rendered:
            # Right under the heading, so the marker stays in the same chunk as
            # the article it refers to.
            position = 1 if rendered[0].startswith("#") else 0
            rendered.insert(position, _pending_note(future))
        body.extend(rendered)

        block_id = block.get("id") or ""
        amending = version.get("id_norma") or ""
        if amending == meta["id"]:
            amending = ""
        index.append(
            {
                "as_of": meta["as_of"],
                "block_id": block_id,
                "kind": block.get("tipo") or "",
                "heading": _heading_of(rendered, block),
                "published": iso_date(version.get("fecha_publicacion")) or "",
                "in_force": iso_date(version.get("fecha_vigencia")) or "",
                "url": f"{meta['source_url']}/texto/bloque/{block_id}",
                "norm_id": meta["id"],
                "consolidation_status": meta["consolidation_status"],
                "source_last_updated": meta["last_updated"],
                "amending_norm": amending,
                "amending_norm_url": (
                    f"https://www.boe.es/buscar/act.php?id={amending}"
                    if amending
                    else ""
                ),
            }
        )

    quote = _notice(meta)

    front = {
        **meta,
        "blocks_pending": len(pending_ids),
        "blocks_with_future_amendment": len(amended_ids),
    }

    document = "\n\n".join(
        [
            _front_matter(front),
            quote,
            f"# {meta['title']}",
            *body,
        ]
    ).rstrip() + "\n"

    stats: dict[str, object] = {
        "blocks_total": len(blocks),
        "blocks_in_force": len(in_force),
        "blocks_expired": len(expired_ids),
        "expired_block_ids": expired_ids,
        "blocks_pending": len(pending_ids),
        "pending_block_ids": pending_ids,
        "blocks_with_future_amendment": len(amended_ids),
        "blocks_with_future_amendment_ids": amended_ids,
        "versions_total": versions_seen,
        "future_versions_total": future_versions,
    }

    return RenderedDocument(
        markdown=document,
        pending_markdown=_pending_document(meta, pending_sections),
        index=index,
        stats=stats,
    )

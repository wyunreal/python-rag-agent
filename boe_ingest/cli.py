"""Download BOE consolidated legislation into data/ and render it as Markdown.

Commands:
    python -m boe_ingest                  # fetch everything in config/norms.toml
    python -m boe_ingest BOE-A-...        # fetch specific norms (registered or not)
    python -m boe_ingest search "penal"   # find a norm's identifier
    python -m boe_ingest list             # what is registered and what is on disk
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from xml.etree import ElementTree as ET

from boe_ingest.boe import (
    BOEError,
    Resource,
    data_of,
    fetch,
    fetch_catalogue,
    parse_catalogue,
)
from boe_ingest.registry import Norm, RegistryError
from boe_ingest.registry import load as load_registry
from boe_ingest.render import REUSE_NOTICE, iso_date, iso_timestamp, render_document

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_REGISTRY = ROOT / "config" / "norms.toml"

SCHEMA_VERSION = 1

LICENSE = (
    "Art. 13 TRLPI: las disposiciones legales y reglamentarias no son objeto "
    "de propiedad intelectual"
)

COMMANDS = {"fetch", "search", "list"}


# --------------------------------------------------------------------------- io


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _text_of(data: ET.Element, path: str) -> str:
    node = data.find(path)
    return (node.text or "").strip() if node is not None else ""


def _parse_day(value: str) -> str:
    """Accept YYYY-MM-DD and return the compact YYYYMMDD the API uses."""
    try:
        return dt.date.fromisoformat(value).strftime("%Y%m%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from exc


def _fold(value: str) -> str:
    """Lowercase and strip accents, so 'Codigo' matches 'Código'."""
    decomposed = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn").lower()


# ------------------------------------------------------------------------- ingest


def _metadata(
    data: ET.Element, norm: Norm, *, registered: bool, as_of: str
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": norm.id,
        "title": _text_of(data, "metadatos/titulo"),
        "rango": _text_of(data, "metadatos/rango"),
        "departamento": _text_of(data, "metadatos/departamento"),
        "ambito": _text_of(data, "metadatos/ambito"),
        "tags": list(norm.tags),
        "registered": registered,
        "as_of": iso_date(as_of),
        "eli_url": _text_of(data, "metadatos/url_eli"),
        "html_url": _text_of(data, "metadatos/url_html_consolidada"),
        "disposition_date": iso_date(_text_of(data, "metadatos/fecha_disposicion")),
        "publication_date": iso_date(_text_of(data, "metadatos/fecha_publicacion")),
        "in_force_date": iso_date(_text_of(data, "metadatos/fecha_vigencia")),
        "last_updated": iso_timestamp(_text_of(data, "metadatos/fecha_actualizacion")),
        "consolidation_status": _text_of(data, "metadatos/estado_consolidacion"),
        "repealed": _text_of(data, "metadatos/estatus_derogacion") == "S",
        "source": "Agencia Estatal BOE - API de legislación consolidada",
        "source_url": f"{Resource(norm.id, 'metadatos').url.rsplit('/', 1)[0]}",
        "license": LICENSE,
        "reuse_notice": REUSE_NOTICE,
    }


def _read_cached(path: Path, *, refresh: bool, resource: Resource) -> bytes:
    if path.exists() and not refresh:
        return path.read_bytes()
    payload = fetch(resource)
    _write_atomic(path, payload)
    return payload


def process(
    norm: Norm,
    *,
    raw_dir: Path,
    corpus_dir: Path,
    refresh: bool,
    registered: bool,
    as_of: str,
) -> dict[str, object]:
    target = raw_dir / norm.id
    metadata_bytes = _read_cached(
        target / "metadatos.xml", refresh=refresh, resource=Resource(norm.id, "metadatos")
    )
    text_bytes = _read_cached(
        target / "texto.xml", refresh=refresh, resource=Resource(norm.id, "texto")
    )

    metadata_data = data_of(metadata_bytes, source=f"{norm.id}/metadatos")
    text_data = data_of(text_bytes, source=f"{norm.id}/texto")

    meta = _metadata(metadata_data, norm, registered=registered, as_of=as_of)
    fetched_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

    rendered = render_document(meta, text_data, as_of=as_of)

    corpus_path = corpus_dir / f"{norm.id}.md"
    _write_atomic(corpus_path, rendered.markdown.encode("utf-8"))

    pending_path = corpus_dir / f"{norm.id}.pending.md"
    if rendered.pending_markdown:
        pending_bytes = rendered.pending_markdown.encode("utf-8")
        _write_atomic(pending_path, pending_bytes)
        pending_entry: dict[str, object] | None = {
            "path": str(pending_path.relative_to(corpus_dir.parent)),
            "sha256": _sha256(pending_bytes),
            "characters": len(rendered.pending_markdown),
        }
    else:
        # A previous run may have written one for another --as-of date.
        pending_path.unlink(missing_ok=True)
        pending_entry = None

    return {
        **meta,
        # Run timestamp lives in the manifest only, so the corpus stays
        # byte-identical when neither the law nor --as-of changed.
        "fetched_at": fetched_at,
        "raw": {
            "metadatos": str((target / "metadatos.xml").relative_to(raw_dir.parent)),
            "metadatos_sha256": _sha256(metadata_bytes),
            "texto": str((target / "texto.xml").relative_to(raw_dir.parent)),
            "texto_sha256": _sha256(text_bytes),
        },
        "corpus": {
            "path": str(corpus_path.relative_to(corpus_dir.parent)),
            "sha256": _sha256(rendered.markdown.encode("utf-8")),
            "characters": len(rendered.markdown),
        },
        "pending": pending_entry,
        "blocks": rendered.stats,
    }


# ----------------------------------------------------------------------- manifest


def _load_manifest(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    entries: dict[str, dict[str, object]] = {}
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if line:
            item = json.loads(line)
            entries[str(item["id"])] = item
    return entries


def _upsert_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    existing = _load_manifest(path)
    existing.update({str(entry["id"]): entry for entry in entries})
    payload = "\n".join(
        json.dumps(existing[key], ensure_ascii=False, sort_keys=True)
        for key in sorted(existing)
    )
    _write_atomic(path, (payload + "\n").encode("utf-8"))


# ---------------------------------------------------------------------- commands


def cmd_fetch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        registry = load_registry(args.registry)
    except RegistryError as exc:
        parser.error(str(exc))

    known = {norm.id: norm for norm in registry}

    if args.ids:
        targets: list[tuple[Norm, bool]] = []
        for norm_id in args.ids:
            norm = known.get(norm_id) or Norm(id=norm_id, title="")
            targets.append((norm, norm_id in known))
    else:
        if not registry:
            parser.error(f"registry is empty: {args.registry}")
        targets = [(norm, True) for norm in registry]

    raw_dir = args.data_dir / "raw"
    corpus_dir = args.data_dir / "corpus"
    as_of = args.as_of or dt.date.today().strftime("%Y%m%d")

    entries: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []

    for norm, registered in targets:
        label = known[norm.id].title if norm.id in known else "(not in registry)"
        print(f"→ {norm.id}  {label}")
        try:
            entry = process(
                norm,
                raw_dir=raw_dir,
                corpus_dir=corpus_dir,
                refresh=args.refresh,
                registered=registered,
                as_of=as_of,
            )
        except BOEError as exc:
            failures.append((norm.id, str(exc)))
            print(f"  failed: {exc}", file=sys.stderr)
            continue

        blocks = entry["blocks"]
        corpus = entry["corpus"]
        pending = blocks["blocks_pending"]
        print(
            f"  ok: {blocks['blocks_in_force']} blocks in force, "
            f"{blocks['blocks_expired']} expired"
            + (f", {pending} not yet in force" if pending else "")
            + f", {corpus['characters']:,} chars → {corpus['path']}"
        )

        if pending:
            print(
                f"  warning: {pending} block(s) excluded, they enter into force "
                f"after {iso_date(as_of)}"
            )
        amended = blocks["blocks_with_future_amendment"]
        if amended:
            print(
                f"  warning: {amended} block(s) are amended by a version that "
                f"enters into force after {iso_date(as_of)}; current wording kept"
            )
        in_force_date = entry.get("in_force_date")
        if in_force_date and in_force_date > iso_date(as_of):
            print(
                f"  warning: the norm itself is not in force until {in_force_date}"
            )

        entries.append(entry)

    if entries:
        _upsert_manifest(args.data_dir / "manifest.jsonl", entries)

    print(f"\n{len(entries)}/{len(targets)} document(s) ready in {args.data_dir}")
    if failures:
        for norm_id, message in failures:
            print(f"  {norm_id}: {message}", file=sys.stderr)
        return 1
    return 0


def cmd_search(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    catalogue_path = args.data_dir / "catalogue.xml"
    # The catalogue is ~11 MB and changes slowly; caching keeps `search` instant.
    if catalogue_path.exists() and not args.refresh_catalogue:
        payload = catalogue_path.read_bytes()
    else:
        payload = fetch_catalogue()
        _write_atomic(catalogue_path, payload)

    entries = parse_catalogue(payload)
    needle = _fold(args.query)

    matches = [entry for entry in entries if needle in _fold(entry.title)]
    if args.active_only:
        matches = [entry for entry in matches if not entry.exhausted]

    def sort_key(entry) -> tuple[int, int, int]:
        """Prefer the norm itself over laws that merely amend it.

        Heuristic, deliberately simple: norms whose validity is exhausted go
        last, amending laws ("... por la que se modifica ...") next, and among
        the rest the shortest title wins, because that is usually the code
        itself rather than a statute about it.
        """
        folded = _fold(entry.title)
        published = int(entry.publication_date) if entry.publication_date.isdigit() else 0
        penalty = 0
        if entry.exhausted:
            penalty += 1000
        if "modifica" in folded:
            penalty += 200
        return (penalty, len(folded), -published)

    matches.sort(key=sort_key)
    shown = matches[: args.limit]

    print(f'{len(matches)} match(es) for "{args.query}" in {len(entries):,} norms\n')
    if not shown:
        return 0

    for entry in shown:
        flag = " (agotada)" if entry.exhausted else ""
        print(
            f"  {entry.id:18} {iso_date(entry.publication_date) or '----------'}  "
            f"{entry.title[:96]}{flag}"
        )

    best = shown[0]
    print("\nAdd this to config/norms.toml:\n")
    print(
        Norm(
            id=best.id,
            title=best.title,
            tags=(_fold(args.query).split()[0],),
        ).to_toml()
    )
    return 0


def cmd_list(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        registry = load_registry(args.registry)
    except RegistryError as exc:
        parser.error(str(exc))

    manifest = _load_manifest(args.data_dir / "manifest.jsonl")

    print(f"Registry: {args.registry}\n")
    if not registry:
        print("  (empty)")
        return 0

    for norm in registry:
        record = manifest.get(norm.id)
        if record:
            corpus = record["corpus"]
            status = f"{corpus['characters']:,} chars, {record['fetched_at']}"
        else:
            status = "not downloaded"
        tags = ",".join(norm.tags) or "-"
        print(f"  {norm.id:18} {tags:16} {status}")
        print(f"  {'':18} {norm.title}")
    return 0


# --------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boe-ingest",
        description="Download BOE consolidated legislation and prepare Markdown.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="download and render norms")
    fetch_parser.add_argument(
        "ids", nargs="*", help="identifiers (default: everything in the registry)"
    )
    fetch_parser.add_argument(
        "--refresh", action="store_true", help="re-download raw XML"
    )
    fetch_parser.add_argument(
        "--as-of",
        type=_parse_day,
        metavar="YYYY-MM-DD",
        help="render the law in force on this date (default: today)",
    )
    fetch_parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="output directory"
    )
    fetch_parser.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY, help="registry file"
    )
    fetch_parser.set_defaults(handler=cmd_fetch)

    search_parser = subparsers.add_parser("search", help="find a norm identifier")
    search_parser.add_argument("query", help="words to look for in titles")
    search_parser.add_argument("--limit", type=int, default=15)
    search_parser.add_argument(
        "--active-only", action="store_true", help="hide norms whose validity is exhausted"
    )
    search_parser.add_argument(
        "--refresh-catalogue", action="store_true", help="re-download the catalogue"
    )
    search_parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="cache directory"
    )
    search_parser.set_defaults(handler=cmd_search)

    list_parser = subparsers.add_parser("list", help="show registry and local status")
    list_parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="output directory"
    )
    list_parser.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY, help="registry file"
    )
    list_parser.set_defaults(handler=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Bare invocation means `fetch`: `python -m boe_ingest --refresh`.
    if not arguments or arguments[0] not in COMMANDS:
        arguments = ["fetch", *arguments]

    parser = build_parser()
    args = parser.parse_args(arguments)

    try:
        return int(args.handler(args, parser))
    except (BOEError, RegistryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 0

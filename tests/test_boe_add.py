from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import boe_ingest.cli as cli
from boe_ingest.boe import CatalogueEntry
from boe_ingest.cli import build_parser, cmd_add


def _entry(norm_id: str, title: str) -> CatalogueEntry:
    return CatalogueEntry(
        id=norm_id,
        title=title,
        rango="Ley",
        ambito="Estatal",
        publication_date="20200101",
        in_force_from="20200102",
        exhausted=False,
        eli_url="",
    )


def _args(registry: Path, *, all: bool = True, limit: int = 15) -> argparse.Namespace:
    return argparse.Namespace(
        query="vivienda",
        limit=limit,
        all=all,
        tag="vivienda",
        active_only=False,
        refresh_catalogue=False,
        dry_run=False,
        registry=registry,
        data_dir=registry.parent,
    )


def test_add_appends_new_norms_and_skips_duplicates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entries = [
        _entry("BOE-A-2020-1", "Ley de vivienda"),
        _entry("BOE-A-2020-2", "Otra ley de vivienda"),
    ]
    monkeypatch.setattr(cli, "_catalogue_entries", lambda data_dir, refresh=False: entries)
    registry = tmp_path / "norms.toml"
    registry.write_text("", encoding="utf-8")

    args = _args(registry)
    assert cmd_add(args, build_parser()) == 0
    text = registry.read_text(encoding="utf-8")
    assert 'id = "BOE-A-2020-1"' in text
    assert 'id = "BOE-A-2020-2"' in text

    assert cmd_add(args, build_parser()) == 0
    text_again = registry.read_text(encoding="utf-8")
    assert text_again.count("BOE-A-2020-1") == 1


def test_add_honours_the_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    entries = [
        _entry("BOE-A-2020-1", "Ley de vivienda"),
        _entry("BOE-A-2020-2", "Otra ley de vivienda"),
    ]
    monkeypatch.setattr(cli, "_catalogue_entries", lambda data_dir, refresh=False: entries)
    registry = tmp_path / "norms.toml"
    registry.write_text("", encoding="utf-8")

    assert cmd_add(_args(registry, all=False, limit=1), build_parser()) == 0
    text = registry.read_text(encoding="utf-8")
    assert 'id = "BOE-A-2020-1"' in text
    assert 'id = "BOE-A-2020-2"' not in text

"""Load the corpus registry (config/norms.toml).

The registry is the single place that defines what belongs to the corpus. The
ingest code never hardcodes identifiers, so adding a norm is a data change.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


class RegistryError(ValueError):
    """Raised when the registry file is missing or malformed."""


@dataclass(frozen=True)
class Norm:
    id: str
    title: str
    tags: tuple[str, ...] = ()
    notes: str = ""

    def to_toml(self) -> str:
        tags = ", ".join(f'"{tag}"' for tag in self.tags)
        return "\n".join(
            [
                "[[norm]]",
                f'id = "{self.id}"',
                f'title = "{self.title}"',
                f"tags = [{tags}]",
            ]
        )


def load(path: Path) -> list[Norm]:
    if not path.exists():
        raise RegistryError(f"registry not found: {path}")

    try:
        payload = tomllib.loads(path.read_text("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise RegistryError(f"{path}: invalid TOML ({exc})") from exc

    entries = payload.get("norm", [])
    if not isinstance(entries, list):
        raise RegistryError(f"{path}: expected a list of [[norm]] tables")

    norms: list[Norm] = []
    seen: set[str] = set()

    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise RegistryError(f"{path}: [[norm]] #{position} is not a table")

        norm_id = str(entry.get("id", "")).strip()
        title = str(entry.get("title", "")).strip()
        if not norm_id or not title:
            raise RegistryError(
                f"{path}: [[norm]] #{position} needs both 'id' and 'title'"
            )
        if norm_id in seen:
            raise RegistryError(f"{path}: duplicate id {norm_id}")
        seen.add(norm_id)

        tags = entry.get("tags", [])
        norms.append(
            Norm(
                id=norm_id,
                title=title,
                tags=tuple(sorted(str(tag) for tag in tags)),
                notes=str(entry.get("notes", "")),
            )
        )

    return norms

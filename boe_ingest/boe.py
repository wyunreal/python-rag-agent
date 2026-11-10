"""HTTP access to the BOE consolidated-legislation open-data API.

Only the officially documented public endpoints are used
(https://www.boe.es/datosabiertos/), with an honest User-Agent and a polite
request rate. There is no authentication, no HTML scraping, and no attempt to
work around any access restriction: if the API refuses a request, the error is
reported to the caller.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from xml.etree import ElementTree as ET

API_ROOT = "https://www.boe.es/datosabiertos/api/legislacion-consolidada"
CATALOGUE_URL = API_ROOT
RESOURCE_ROOT = f"{API_ROOT}/id"

USER_AGENT = (
    "python-rag-agent/0.1 "
    "(BOE open-data ingestion; https://www.boe.es/datosabiertos/)"
)

TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 3
MIN_INTERVAL_SECONDS = 1.0

_last_request_at = 0.0


class BOEError(RuntimeError):
    """Raised when a BOE API resource cannot be fetched or understood."""


@dataclass(frozen=True)
class Resource:
    """A single documented endpoint: /id/{norm_id}/{kind}."""

    norm_id: str
    kind: str  # "metadatos" or "texto"

    @property
    def url(self) -> str:
        return f"{RESOURCE_ROOT}/{self.norm_id}/{self.kind}"


@dataclass(frozen=True)
class CatalogueEntry:
    """One row of the consolidated-norms catalogue, used by `search`."""

    id: str
    title: str
    rango: str
    ambito: str
    publication_date: str
    in_force_from: str
    exhausted: bool
    eli_url: str


def _throttle() -> None:
    """Keep at least MIN_INTERVAL_SECONDS between requests, politely."""
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < MIN_INTERVAL_SECONDS:
        time.sleep(MIN_INTERVAL_SECONDS - elapsed)
    _last_request_at = time.monotonic()


def _retry_after(header: str | None, attempt: int) -> float:
    """Honour Retry-After when the server sends it, capped to a sane wait."""
    if header:
        try:
            return min(max(float(header), 0.0), 60.0)
        except ValueError:
            pass
    return float(2 ** (attempt - 1))


def _get(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/xml", "User-Agent": USER_AGENT},
    )

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            _throttle()
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError):
                if exc.code in (429, 503):
                    # Back off as asked, then retry.
                    if attempt < MAX_ATTEMPTS:
                        time.sleep(_retry_after(exc.headers.get("Retry-After"), attempt))
                        continue
                    break
                # Any other refusal is a decision, not a hiccup.
                if 400 <= exc.code < 500:
                    break
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))

    raise BOEError(f"GET {url} failed: {last_error}")


def fetch(resource: Resource) -> bytes:
    """GET one norm resource and return its raw XML bytes."""
    return _get(resource.url)


def fetch_catalogue() -> bytes:
    """GET the full list of consolidated norms (one 11 MB request)."""
    return _get(f"{CATALOGUE_URL}?limit=-1")


def data_of(xml_bytes: bytes, *, source: str) -> ET.Element:
    """Validate the API envelope and return its <data> element."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise BOEError(f"{source}: malformed XML ({exc})") from exc

    code = (root.findtext("status/code") or "").strip()
    if code != "200":
        message = (root.findtext("status/text") or "").strip()
        raise BOEError(f"{source}: API returned status {code!r} ({message})")

    data = root.find("data")
    if data is None:
        raise BOEError(f"{source}: response contains no <data> element")
    return data


def parse_catalogue(xml_bytes: bytes) -> list[CatalogueEntry]:
    """Parse the catalogue response into rows."""
    data = data_of(xml_bytes, source="catálogo")

    def text(item: ET.Element, tag: str) -> str:
        node = item.find(tag)
        return (node.text or "").strip() if node is not None else ""

    entries: list[CatalogueEntry] = []
    for item in data.iter("item"):
        norm_id = text(item, "identificador")
        if not norm_id:
            continue
        entries.append(
            CatalogueEntry(
                id=norm_id,
                title=text(item, "titulo"),
                rango=text(item, "rango"),
                ambito=text(item, "ambito"),
                publication_date=text(item, "fecha_publicacion"),
                in_force_from=text(item, "fecha_vigencia"),
                exhausted=text(item, "vigencia_agotada") == "S",
                eli_url=text(item, "url_eli"),
            )
        )
    return entries

# boe_ingest

Download and prepare Spanish consolidated legislation from the BOE open-data
API. This package is the ingest stage: it fetches the norm, selects the version
in force, and renders Markdown. No chunking, embeddings or retrieval.

Part of [python-rag-agent](../README.md).

## Commands

| Command | Effect |
|---|---|
| `python -m boe_ingest` | fetch and render every registered norm |
| `python -m boe_ingest <ID>…` | fetch specific norms, registered or not |
| `python -m boe_ingest --refresh` | ignore the raw cache and re-download |
| `python -m boe_ingest --as-of 1990-01-01` | render the law in force on a date |
| `python -m boe_ingest list` | registry and local status |
| `python -m boe_ingest search "<text>"` | find identifiers |

## Adding a norm

```bash
python -m boe_ingest search "estatuto de los trabajadores"
```

Paste the `[[norm]]` snippet it prints into `config/norms.toml`, then run
`python -m boe_ingest`. Only the new norm hits the network; the rest re-render
from cache.

To try one without registering it, pass the ID directly — its manifest entry
records `registered: false`. To remove a norm, delete its `[[norm]]` table.

`search` filters a cached catalogue (`data/catalogue.xml`, ~11 MB, 10,000 norms)
because the API's own `query` parameter returns HTTP 500.

## Output

```
data/
├── raw/<ID>/{metadatos,texto}.xml   # verbatim API responses
├── corpus/<ID>.md                   # law in force as of the reference date
├── corpus/<ID>.pending.md           # only when future wording exists
├── catalogue.xml                    # cached list for `search`
└── manifest.jsonl                   # provenance, hashes, block counts
```

`data/` is gitignored and fully regenerable.

Runs are idempotent: raw XML survives (unless `--refresh`), the corpus is
rewritten atomically, and a stale `<ID>.pending.md` is deleted. `fetched_at`
lives only in the manifest, so the rendered files are byte-identical when
neither the law nor `--as-of` changed.

## How it works

1. Two requests per norm: `/id/{ID}/metadatos` and `/id/{ID}/texto`.
2. The `/texto` response contains **every historical version** of every block
   (3,837 for the Civil Code). The one rendered is the latest whose
   `fecha_vigencia` is not later than the reference date.
3. Blocks carrying `fecha_caducidad` are repealed: dropped from the body, their
   IDs recorded in the manifest (167 Civil Code, 36 Penal Code).
4. Structure comes from the `<p class="…">` attributes, so headings are read
   from the source rather than inferred.
5. Raw XML is never modified, so extractor fixes never require re-downloading.
   Every raw file and rendered document is SHA-256 hashed.

Access is polite: honest `User-Agent`, 1 request/second, `4xx` reported and not
retried. No HTML scraping.

## Markdown structure

| Level | Markdown |
|---|---|
| Libro | `##` |
| Título | `###` |
| Capítulo | `####` |
| Sección / Subsección | `#####` |
| Artículo | `######` |

Front-matter carries `id`, `title`, `rango`, `tags`, `eli_url`, dates,
`consolidation_status`, `as_of`, `blocks_pending`,
`blocks_with_future_amendment` and `schema_version`.

## Dates and versions

### `--as-of`

Everything is rendered relative to a reference date: today by default, or any
date given as `YYYY-MM-DD`.

```bash
python -m boe_ingest --as-of 1990-01-01
```

It decides:

- which version of each block is rendered — the latest whose `fecha_vigencia` is
  not later than that date;
- which blocks count as repealed (`fecha_caducidad`);
- which blocks are not yet in force;
- whether `<ID>.pending.md` is generated;
- the `as_of` value written to the front-matter and the manifest.

It does **not** change what is downloaded. `data/raw/` always holds every
version, so changing the date re-renders from cache without touching the network.

Same norm, two dates:

| Código Civil | In force | Repealed | Not yet in force | With later amendment |
|---|---|---|---|---|
| today | 2,277 | 167 | 0 | 0 |
| 1990-01-01 | 2,321 | 84 | 39 | 439 |

This is what makes temporal questions answerable: `--as-of 1995-01-01` renders
the wording in force that day. An invalid date is rejected before any work is
done.

### Pending wording

| Situation | Effect |
|---|---|
| Block whose every version enters into force later | Excluded from `<ID>.md`, written to `<ID>.pending.md` |
| Block with current wording and a later version | Current wording kept, marked inline, future text in `<ID>.pending.md` |
| Norm not yet in force at all | Reported; `<ID>.md` carries only its header |

The marker sits right under the article heading, so it stays in the same chunk a
retriever will see:

```
###### Artículo 8.

> **Reforma pendiente** — nueva redacción con efectos desde 2001-01-08, publicada el 2000-01-08. No vigente todavía.
```

`<ID>.pending.md` holds the actual future wording, kept out of the main document
so the corpus stays "law in force". Whether to index it is a retrieval decision.

These are reforms **already approved and published** whose entry into force is
deferred. Bills still in parliament are not part of the BOE consolidated text, so
this corpus cannot answer questions about them.

The manifest records `blocks_pending` and `blocks_with_future_amendment`, plus
the pending file under `pending` with its own hash.

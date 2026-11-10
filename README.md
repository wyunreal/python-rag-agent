# python-rag-agent

RAG over Spanish consolidated legislation via the BOE open-data API.
Current state: download and prepare only — no chunking, embeddings or retrieval yet.

## Corpus

Defined in [`config/norms.toml`](config/norms.toml), not in code.

| Document | BOE ID | Tag |
|---|---|---|
| Constitución Española | `BOE-A-1978-31229` | `constitucional` |
| Código Civil | `BOE-A-1889-4763` | `civil` |
| Código Penal (LO 10/1995) | `BOE-A-1995-25444` | `penal` |

## Setup

Python 3.11+, no dependencies.

```bash
make venv                 # create .venv
source .venv/bin/activate
make list                 # sanity check
```

`make clean` removes the venv and refuses to delete a non-venv directory.
`.venv/` is gitignored: it is build output, the `Makefile` is code.

## Documentation

- [`boe_ingest/README.md`](boe_ingest/README.md) — commands, how the download
  works, corpus layout, adding norms, `--as-of` and pending amendments.
- [`config/norms.toml`](config/norms.toml) — the corpus registry.

## Data and legal notice

The **code is MIT** (see [`LICENSE`](LICENSE)); that licence does not extend to
the legal texts.

- Texts come from the BOE open-data API. Legal provisions are excluded from
  copyright under **Art. 13 TRLPI** (RDL 1/1996).
- Reuse follows the BOE's conditions (framework: **Ley 37/2007**, **RD
  1495/2011**): attribute the source, do not distort the content, keep the
  update dates, do not imply official endorsement.
- Every generated document carries that attribution, the reference date, the
  official URL and the notice that the authentic version is the one published in
  the BOE.
- The corpus is generated locally and is **not** distributed here: `data/` is
  gitignored and fully regenerable.
- This project is independent from the BOE and is not endorsed by it.

Art. 13 covers legal provisions and court rulings, not every document published
in an official gazette. Any new source needs its own licence and reuse analysis
before joining the corpus.

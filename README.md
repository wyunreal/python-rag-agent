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
- Reuse follows the BOE's **conditions of reuse** (Resolución of 27 June 2024,
  in force since 28 June 2024): do not distort the content, cite the source with
  a link to <https://www.boe.es>, keep the update dates, identify any
  modification, and do not suggest official character or endorsement.
- Because the documents are transformed, the citation used is the one the
  licence requires for derivative reuse — *«Basado en datos de la Agencia Estatal
  Boletín Oficial del Estado»* — plus the adaptation note.
- Each generated document also states that a consolidated text is merely
  informative, and gives the BOE identifier, the last update published by the
  BOE, the reference date and the official URL.
- The corpus is generated locally and is **not** distributed here: `data/` is
  gitignored and fully regenerable.
- This project is independent from the BOE and is not endorsed by it.

Art. 13 covers legal provisions and court rulings, not every document published
in an official gazette. Any new source needs its own licence and reuse analysis
before joining the corpus.

**On temporal accuracy:** `--as-of` rebuilds versions from the consolidated text,
which reflects express amendments and repeals. It does not resolve tacit
repeals, transitional regimes or interpretive questions, and it is not a
substitute for checking the official publication. When the BOE flags a text as
*Desactualizado*, the generated document says so visibly.

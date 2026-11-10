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

Legal texts are not subject to intellectual property (**Art. 13 TRLPI**). Each
output file states that a consolidated text is *meramente informativo* and that
the authentic version is the one published in the BOE.

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

## License

Code is MIT (see [`LICENSE`](LICENSE)). `data/` is gitignored and fully
regenerable, so the repository ships code and configuration only.

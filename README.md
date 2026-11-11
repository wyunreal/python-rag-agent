# python-rag-agent

RAG over Spanish consolidated legislation via the BOE open-data API.
The pipeline downloads and renders the corpus, indexes it with local embeddings
and answers questions with a ReAct agent built on LangChain.

## Corpus

Defined in [`config/norms.toml`](config/norms.toml), not in code.

| Document | BOE ID | Tag |
|---|---|---|
| Constitución Española | `BOE-A-1978-31229` | `constitucional` |
| Código Civil | `BOE-A-1889-4763` | `civil` |
| Código Penal (LO 10/1995) | `BOE-A-1995-25444` | `penal` |

## Setup

Python 3.11+.

```bash
make venv                 # create .venv
source .venv/bin/activate
pip install -e ".[dev]"   # runtime deps + dev tools
make list                 # sanity check
```

The first `pip install` pulls `sentence-transformers` and PyTorch (needed to run
`bge-m3` locally). Development tools live in the `dev` extra.

```bash
make lint                 # ruff check over boe_ingest, agent and tests
make format               # ruff format
```

`make clean` removes the venv and refuses to delete a non-venv directory.
`.venv/` is gitignored: it is build output, the `Makefile` is code.

## Agent

[`agent/`](agent/) is a ReAct agent built with LangChain's `create_agent`
(the prebuilt ReAct loop in LangChain 1.x). It searches the corpus before
answering and cites the norm, article and official BOE URL.

- **LLM:** Ollama Cloud through its OpenAI-compatible endpoint (`/v1`). The
  default is `deepseek-v4.1-flash:1017` (1M context, strong tool calling).
  `glm-5.3-flash` is faster and cheaper; `gpt-oss:120b` is older and may be
  retired. Change it with `OLLAMA_MODEL`.
- **API:** the OpenAI Responses API (`POST /v1/responses`) in stateless mode:
  the whole message history is sent on every request and no
  `previous_response_id` is used. Reasoning items are requested with
  `include=["reasoning.encrypted_content"]`, kept in the message history and
  sent back on later turns (including tool-call turns). Set
  `AGENT_API_MODE=chat_completions` to go back to `/v1/chat/completions`.
- **Embeddings:** `BAAI/bge-m3`, run locally. Ollama Cloud does not serve
  embedding models, and bge-m3 is strong on Spanish and multilingual text.
  The first run downloads about 2 GB; a GPU is used when available.
- **Vector store:** `data/index/` (`embeddings.npy` + `chunks.jsonl`). It is
  rebuilt automatically when the corpus, the embedding model or the chunking
  version changes.

```bash
cp .env.example .env      # add OLLAMA_API_KEY from ollama.com/settings/keys
make index                # build data/index (or: python -m agent index)

python -m agent ask "¿Cuál es el plazo máximo de detención preventiva?"
python -m agent chat      # multi-turn
python -m agent search "mayoría de edad" -k 3 --norm BOE-A-1889-4763
python -m agent info
```

`python -m agent "..."` is shorthand for `ask`; `search` exercises retrieval
without calling the LLM. Configuration lives in environment variables, documented
in [`.env.example`](.env.example).

How the pieces fit:

1. [`agent/corpus.py`](agent/corpus.py) splits each Markdown document on its
   heading hierarchy, keeps the enclosing book/title/chapter as metadata and
   only breaks long articles into overlapping pieces.
2. [`agent/index.py`](agent/index.py) embeds the chunks, persists a numpy matrix
   and loads it lazily. A fingerprint of the corpus files makes staleness cheap
   to detect.
3. [`agent/tools.py`](agent/tools.py) exposes `search_corpus`, which returns
   fragments with their article, section, validity date and URL.
4. [`agent/agent.py`](agent/agent.py) builds the LangChain agent with the model,
   the tool and the legal system prompt in
   [`agent/prompts.py`](agent/prompts.py).

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
- No BOE logo or corporate image is used, and nothing suggests official status.
- The reuse conditions do **not** cover the BOE's *Biblioteca Jurídica Digital*
  (CC BY-NC-ND 4.0), nor site design, logos or third-party material. None of
  those are in the corpus.
- This project is independent from the BOE and is not endorsed by it.

### What a generated document looks like

Every file starts with this block, before any legal text:

```
> Basado en datos de la [Agencia Estatal Boletín Oficial del Estado](https://www.boe.es).
>
> Texto consolidado de carácter meramente informativo. La versión auténtica es la publicada en el «Boletín Oficial del Estado».
>
> Documento: BOE-A-1889-4763 · última actualización 2026-09-07 · referencia 2026-11-10
> https://www.boe.es/buscar/act.php?id=BOE-A-1889-4763
>
> Formato y estructura adaptados a Markdown por python-rag-agent.
> Proyecto independiente del BOE; no está respaldado por él.
```

The pending-amendments file (`<ID>.pending.md`) carries the same block plus an
explicit warning that its content is not in force.

Art. 13 covers legal provisions and court rulings, not every document published
in an official gazette. Any new source needs its own licence and reuse analysis
before joining the corpus.

**On temporal accuracy:** `--as-of` rebuilds versions from the consolidated text,
which reflects express amendments and repeals. It does not resolve tacit
repeals, transitional regimes or interpretive questions, and it is not a
substitute for checking the official publication. When the BOE flags a text as
*Desactualizado*, the generated document says so visibly.

## Before serving answers

None of this applies to local, single-user experiments, but it is required
before exposing a public Q&A service:

- Cite article, norm and normative reference date in every answer, with a link
  to the official publication.
- Surface stale consolidation and pending amendments instead of presenting them
  as current law.
- State that answers are informative, not legal advice, and that the official
  text must be checked before acting on them.
- RGPD/LOPDGDD: legal basis, privacy notice, processors, retention and
  international transfers — including what the model provider does with prompts.
- EU AI Act: transparency obligations for systems interacting with people
  (disclose that the user is talking to an AI system).
- Terms of use and a contact address for corrections.

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

Python 3.11+. Docker and Docker Compose are needed for the default embedding
server; Langfuse is optional.

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

## Quick start (all local)

The default path uses the persistent embedding server:

```bash
cp .env.example .env       # add OLLAMA_API_KEY
python -m boe_ingest       # download and render data/corpus (first run)
make embed-up              # required by the default .env
make index                 # build data/index
python -m agent ask "¿Cuál es el plazo máximo de detención preventiva?"
```

Optional Langfuse tracing:

```bash
make langfuse-up           # Langfuse UI on localhost:3000
# uncomment LANGFUSE_* in .env
python -m agent chat
```

If you prefer not to run Docker for embeddings, comment out
`AGENT_EMBED_SERVER` in `.env`; the agent will load `bge-m3` inside the process
(about 5 s in the first `ask`). Without Langfuse keys, tracing is disabled.

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
- **Embeddings:** `BAAI/bge-m3`. By default queries go to the persistent
  embedding server in [`embed/`](embed/) (CPU). If `AGENT_EMBED_SERVER` is unset,
  the model runs locally on CPU; set `AGENT_DEVICE=cuda` for the local path. The
  first local run downloads about 2 GB. Ollama Cloud does not serve embedding
  models, and bge-m3 is strong on Spanish and multilingual text.
- **Vector store:** `data/index/` (`embeddings.npy` + `chunks.jsonl`). It is
  rebuilt automatically when the corpus, the embedding model or the chunking
  version changes.

```bash
cp .env.example .env      # add OLLAMA_API_KEY from ollama.com/settings/keys
python -m boe_ingest      # download and render data/corpus
make embed-up             # embedding server enabled by default
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
   and keeps it in memory. A fingerprint of the corpus files makes staleness
   cheap to detect; repeated tool calls avoid reading the files again. Query
   embeddings are cached per process too, so asking the same question twice
   skips the embedding model.
3. [`agent/tools.py`](agent/tools.py) exposes `search_corpus`, which returns
   fragments with their article, section, validity date and URL.
4. [`agent/agent.py`](agent/agent.py) builds the LangChain agent with the model,
   the tool and the legal system prompt in
   [`agent/prompts.py`](agent/prompts.py).

## Tracing with Langfuse (local)

[`langfuse/docker-compose.yml`](langfuse/docker-compose.yml) runs a local
Langfuse v4 stack: web, worker, Postgres, ClickHouse, Redis and MinIO. It is
intended for local development only and uses fixed credentials.

```bash
make langfuse-up       # first run pulls images
make langfuse-logs     # optional
make langfuse-down     # stop; volumes are kept
```

- UI: <http://localhost:3000>
- Login: `admin@local.test` / `langfuse-local`
- Project keys: `pk-lf-local` / `sk-lf-local`

Uncomment these variables in `.env` after `make langfuse-up`:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-local
LANGFUSE_SECRET_KEY=sk-lf-local
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_PROJECT_ID=local-project
```

With the keys present, `ask` sends one trace per invocation and `chat` groups
all turns of a conversation under one Langfuse session. Each run prints the
trace URL; pass `--timings` to `ask` or `chat` for a console summary
(`first_llm`, `tool_done`, `first_answer`, `total`). Inside a trace,
`search_corpus` contains `retrieval` and its children `ensure_index`,
`embed_query`, `vector_search` and `format_hits`, each with its own latency.
Every model call also has an `llm_call` child whose input is the exact API
payload (Responses or Chat Completions format) and whose output is the generated
message with reasoning, tool calls and text.
Set `LANGFUSE_TRACING_ENABLED=false` to disable tracing without removing the
keys.

The stack stores its data in Docker volumes. `make langfuse-down` keeps them;
use `docker compose -f langfuse/docker-compose.yml down -v` to start from
scratch. Do not expose port 3000 publicly with these credentials.

## Persistent embedding server

Loading `bge-m3` costs about 5 s per process: importing torch/transformers
(~3.4 s) plus reading the weights (~1.5 s). [`embed/`](embed/) keeps the model
warm in a Docker container so `ask` does not pay that cost every time.

```bash
make embed-up        # build and start; first build is slow
make embed-logs
make embed-down
```

The service listens on `127.0.0.1:8765` and exposes `POST /embed`.
`.env.example` enables it by default:

```env
AGENT_EMBED_SERVER=http://127.0.0.1:8765
```

Run `make embed-up` before `ask`, `chat` or `search`. If the container is down,
those commands exit at startup with a clear error. Comment out
`AGENT_EMBED_SERVER` to load the model locally instead. The container runs on
CPU, reuses `${HOME}/.cache/huggingface`, and only serves query embeddings;
index builds still run on the host.

### Deploying in a cluster

The agent reads service URLs from environment variables; nothing assumes
`localhost`:

```env
LANGFUSE_BASE_URL=https://langfuse.example.com
LANGFUSE_PROJECT_ID=<project-id>
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
AGENT_EMBED_SERVER=http://embed-server:8765
```

The `langfuse/` and `embed/` compose files are local conveniences. In a cluster,
point `AGENT_EMBED_SERVER` at the embedding Service and `LANGFUSE_BASE_URL` at
the Langfuse ingress. For the Langfuse container itself, set `NEXTAUTH_URL` to
its public URL; `LANGFUSE_PORT` and `EMBED_PORT` change the host port mappings
when using Compose.

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

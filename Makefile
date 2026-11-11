# Development helpers.
#
# The virtual environment itself is NOT versioned: it is build output, machine
# specific and reproducible from this file. These targets are code, so they are.

VENV   ?= .venv
PYTHON ?= python3
PY     := $(VENV)/bin/python
COMPOSE := docker compose -f langfuse/docker-compose.yml --env-file langfuse/.env
EMBED_COMPOSE := docker compose -f embed/docker-compose.yml --env-file .env

.PHONY: help venv clean recreate list index info lint format \
        langfuse-up langfuse-down langfuse-logs \
        embed-up embed-down embed-logs

help:
	@echo "make venv          create $(VENV) if it does not exist"
	@echo "make clean         remove $(VENV)"
	@echo "make recreate      remove and recreate $(VENV)"
	@echo "make list          show the registered corpus"
	@echo "make index         build the agent embedding index"
	@echo "make info          show agent configuration and index status"
	@echo "make lint          run ruff over boe_ingest, agent and tests"
	@echo "make format        run ruff format over boe_ingest, agent and tests"
	@echo "make langfuse-up   start local Langfuse (docker compose)"
	@echo "make langfuse-down stop local Langfuse"
	@echo "make langfuse-logs follow Langfuse logs"
	@echo "make embed-up      build and start the persistent embedding server"
	@echo "make embed-down    stop the embedding server"
	@echo "make embed-logs    follow embedding server logs"
	@echo
	@echo "Override the location with: make venv VENV=/tmp/other-venv"

venv: $(PY)

$(PY):
	$(PYTHON) -m venv $(VENV)
	@echo "Created $(VENV)"
	@echo "Activate it with: source $(VENV)/bin/activate"

clean:
	@if [ -f "$(VENV)/pyvenv.cfg" ]; then \
		rm -rf "$(VENV)" && echo "Removed $(VENV)"; \
	else \
		echo "'$(VENV)' is not a virtualenv; refusing to delete it" >&2; \
		exit 1; \
	fi

recreate: clean venv

list: venv
	$(PY) -m boe_ingest list

index: venv
	$(PY) -m agent index

info: venv
	$(PY) -m agent info

langfuse-up:
	@test -f langfuse/.env || cp langfuse/.env.example langfuse/.env
	$(COMPOSE) up -d --wait

langfuse-down:
	$(COMPOSE) down

langfuse-logs:
	$(COMPOSE) logs -f

embed-up:
	$(EMBED_COMPOSE) up -d --build

embed-down:
	$(EMBED_COMPOSE) down

embed-logs:
	$(EMBED_COMPOSE) logs -f

lint: venv
	@command -v $(VENV)/bin/ruff >/dev/null || { \
		echo "ruff not installed. Run: $(PY) -m pip install -e '.[dev]'" >&2; exit 1; }
	@$(PY) -m ruff check boe_ingest agent tests

format: venv
	@command -v $(VENV)/bin/ruff >/dev/null || { \
		echo "ruff not installed. Run: $(PY) -m pip install -e '.[dev]'" >&2; exit 1; }
	@$(PY) -m ruff format boe_ingest agent tests

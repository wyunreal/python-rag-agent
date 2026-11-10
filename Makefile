# Development helpers.
#
# The virtual environment itself is NOT versioned: it is build output, machine
# specific and reproducible from this file. These targets are code, so they are.

VENV   ?= .venv
PYTHON ?= python3
PY     := $(VENV)/bin/python

.PHONY: help venv clean recreate list

help:
	@echo "make venv      create $(VENV) if it does not exist"
	@echo "make clean     remove $(VENV)"
	@echo "make recreate  remove and recreate $(VENV)"
	@echo "make list      show the registered corpus"
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

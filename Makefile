PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
DATASET ?= data/command-safety/processed/development.jsonl
MANIFEST ?= tasks/command-safety-v1.yaml
ARTIFACT_DIR ?= artifacts/command-safety-v1
MODEL ?= gpt-5.6-terra:medium

.PHONY: help setup validate-data train test infer go-test generate clean

help:
	@printf '%s\n' \
	  'make setup                         Create the Python environment and install test dependencies.' \
	  'make validate-data DATASET=<path>  Validate and normalize JSONL records.' \
	  'make generate                      Generate a synthetic raw JSONL batch with std_slop.' \
	  'make train DATASET=<path>           Train and export a model artifact.' \
	  'make test                          Run Python tests.' \
	  'make go-test                       Run Go tests.' \
	  'make infer TEXT=<command>           Classify without executing TEXT.'

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e '.[test]'

validate-data:
	$(PY) -m commandclassifier.validate_data --input $(DATASET)

generate:
	@mkdir -p data/command-safety/raw
	std_slop --model $(MODEL) --prompt_file prompts/command-safety-v1.md --output json | $(PY) -c 'import json, sys; print(json.load(sys.stdin)["assistant_message"])' > data/command-safety/raw/std-slop-batch-001.jsonl

train:
	$(PY) -m commandclassifier.train --manifest $(MANIFEST) --input $(DATASET) --artifact-dir $(ARTIFACT_DIR)

test:
	$(PY) -m pytest

go-test:
	go test ./...

infer:
	$(PY) -m commandclassifier.cli --artifact-dir $(ARTIFACT_DIR) --text '$(TEXT)'

clean:
	rm -rf .venv .pytest_cache build dist

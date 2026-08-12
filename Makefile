PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
DATASET ?= data/command-safety/processed/development.jsonl
MANIFEST ?= tasks/command-safety-v1.yaml
ARTIFACT_DIR ?= artifacts/command-safety-v1
EVALUATION ?= data/command-safety/reviewed/locked-eval-source-disjoint-v1.jsonl
MODEL ?= gpt-5.6-terra:medium
TASK ?= $(MANIFEST)
BACKEND ?= std_slop
BATCH ?= 001
FOCUS ?= balanced Linux and macOS command safety examples
SESSION ?= commandclassifier-batch-$(BATCH)
GO_ARTIFACT ?= $(ARTIFACT_DIR)/model.json
GO_BIN ?= $(CURDIR)/bin

.PHONY: help setup validate-data corpus-report train test quality infer go-infer go-install go-test generate clean

help:
	@printf '%s\n' \
	  'make setup                         Create the Python environment and install test dependencies.' \
	  'make validate-data DATASET=<path>  Validate and normalize JSONL records.' \
	  'make generate TASK=<manifest> BACKEND=claude|std_slop BATCH=002 FOCUS=... MODEL=<model> Generate raw candidate material.' \
	  'make train DATASET=<path>           Verify corpus separation, train, and export an artifact.' \
	  'make test                          Run Python unit and property tests.' \
	  'make quality                       Check Python formatting and lint rules.' \
	  'make go-test                       Run Go unit tests.' \
	  'make go-infer TEXT=<command>        Classify with the portable Go artifact.' \
	  'make go-install                     Install command-classify to ./bin.' \
	  'make infer TEXT=<command>           Classify without executing TEXT.'

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e '.[test]'

validate-data:
	$(PY) -m commandclassifier.validate_data --manifest $(MANIFEST) --input $(DATASET)

corpus-report:
	$(PY) -m commandclassifier.corpus_report --manifest $(MANIFEST) --development $(DATASET) --evaluation $(EVALUATION) --output reports/corpus-report.json

generate:
	$(PY) -m commandclassifier.generate --task $(TASK) --backend $(BACKEND) --batch $(BATCH) --focus '$(FOCUS)' --model '$(MODEL)'

train:
	$(PY) -m commandclassifier.train --manifest $(MANIFEST) --input $(DATASET) --evaluation $(EVALUATION) --artifact-dir $(ARTIFACT_DIR)

test:
	$(PY) -m pytest

quality:
	$(PY) -m ruff format --check .
	$(PY) -m ruff check .

go-test:
	cd go && go test ./...

go-infer:
	cd go && go run ./cmd/command-classify --model ../$(GO_ARTIFACT) --text '$(TEXT)'

go-install:
	@mkdir -p $(GO_BIN)
	cd go && GOBIN=$(GO_BIN) go install ./cmd/command-classify

infer:
	$(PY) -m commandclassifier.cli --artifact-dir $(ARTIFACT_DIR) --text '$(TEXT)'

clean:
	rm -rf .venv .pytest_cache build dist

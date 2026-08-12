# Reviewed Binary Text Classification Framework

This is a framework for building **reviewed synthetic binary text-classification tasks**: define a task once, generate and review its data, train a calibrated classifier, and ship a single portable artifact that runs identically in Python and Go.
The framework classifies text only. It never executes supplied text.

## What it does

A task is declared in a manifest (see `templates/binary-task/`): one or more text input fields, exactly two labels, one positive class, split-group fields, and a decision/review probability policy. Training data is synthetic but human-reviewed; the framework never trusts generated output directly.

The pipeline:

1. **Generate** — a backend (`std_slop` or `claude -p`) produces raw candidate records from the task prompt, preserved verbatim for audit.
2. **Review** — every record is reviewed before it enters a development or locked-evaluation dataset.
3. **Train** — grouped nested evaluation, calibration, and threshold selection produce a v3 artifact. Training rejects duplicate canonical inputs and development/locked-evaluation overlap.
4. **Infer** — Python and Go load the same portable artifact, verify its hash and contract, and classify with identical input serialization.

The artifact (`model.json` plus `model.json.sha256`; Python also writes `model.joblib`) bundles task metadata, input serialization, labels, probability output name, policy, feature and calibration metadata, and input hashes. It is the only file a consumer needs.

## Quick start

Requirements: Python 3.11+, Go, and `make`. Set up and verify the environment once:

```bash
make setup
make quality
make test
```

The repository ships two classifiers, each with its own manifest, reviewed datasets, and artifact:

- `command-safety-v1` — classifies shell-command text as `safe` or `unsafe` (`unsafe` is positive). Advisory only: a `safe` result is not permission to run a command, and the classifier never runs it.
- `prompt-complexity-v1` — classifies a user prompt to an LLM as `low_complexity` or `high_complexity` (`high_complexity` is positive). Terse continuations such as `continue` and `proceed` are low-complexity by design.

Both use the same v3 artifact contract and run identically in Python and Go. `make infer`/`make go-infer` point at an artifact via `ARTIFACT_DIR`; `make train` points at a task via `ARTIFACT_DIR`, `MANIFEST`, `DATASET`, and `EVALUATION`.

### Train a classifier

`make train` requires a reviewed, hash-bound locked-evaluation manifest and rejects development/evaluation input overlap before writing an artifact. The default targets command-safety:

```bash
make train                             # command-safety (default)

make train \
  ARTIFACT_DIR=artifacts/prompt-complexity-v1 \
  MANIFEST=tasks/prompt-complexity-v1.yaml \
  DATASET=data/prompt-complexity/processed/development.jsonl \
  EVALUATION=data/prompt-complexity/reviewed/locked-eval-v1.jsonl   # prompt-complexity
```

### Infer on either classifier

```bash
# command-safety (default artifact)
make infer TEXT='git status'
make go-infer TEXT='git status'

# prompt-complexity
make infer ARTIFACT_DIR=artifacts/prompt-complexity-v1 TEXT='continue'
make go-infer ARTIFACT_DIR=artifacts/prompt-complexity-v1 TEXT='continue'
```

The Python and Go commands load the same artifact, verify its hash and contract, and produce identical `label`, `positive_probability`, `confidence`, `review_recommended`, and `model_version` output.

## Codebase layout

```text
src/generic_binary_classifier/   Python pipeline and CLI (generate, validate, merge, train, export, infer)
go/                      Go inference runtime using the same v3 artifact
templates/binary-task/   Scaffolding for a new task
tasks/ prompts/ schemas/ Per-task manifests, generation prompts, record schemas
data/                    Raw, processed, and reviewed datasets per task
artifacts/               Exported model artifacts
docs/                    Model cards, release criteria, framework details
tests/                   Python unit and property test suite
```

## Learn more

- [Framework details](docs/details.md) — task definition, full commands, release checklist
- [Task creation guide](docs/task-creation/SKILL.md) — how to add a new task
- [Command-safety model card](docs/model-card.md) and [release criteria](docs/release-criteria.md)
- [Prompt-complexity model card](docs/prompt-complexity-model-card.md) and [release criteria](docs/prompt-complexity-release-criteria.md)

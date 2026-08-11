# Reviewed Binary Text Classification Framework

This is a  framework for building **reviewed synthetic binary text-classification tasks**: define a task once, generate and review its data, train a calibrated classifier, and ship a single portable artifact that runs identically in Python and Go.
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

Requirements: Python 3.11+, Go, and `make`.

```bash
make setup
make quality
make test
make train
make infer TEXT='git status'
make go-infer TEXT='git status'
```

The included `command-safety-v1` task classifies shell-command text as `safe` or `unsafe` (`unsafe` is positive). Its result is advisory only: a `safe` result is not permission to run a command, and the classifier never runs it. `make train` requires a reviewed, hash-bound locked-evaluation manifest and rejects development/evaluation input overlap before writing an artifact.

## Codebase layout

```text
src/commandclassifier/   Python pipeline and CLI (generate, validate, merge, train, export, infer)
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
- [Model card](docs/model-card.md), [release criteria](docs/release-criteria.md), and [external evaluation evidence](docs/external-evaluation-evidence.md) for the included task

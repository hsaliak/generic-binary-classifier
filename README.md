# Command Classifier

Command Classifier is a reusable framework for training versioned text classifiers from reviewed datasets. The first task, `command-safety-v1`, classifies Linux and macOS shell-command text as `safe` or `unsafe` and returns a calibrated unsafe probability.

The Python and Go tools classify text only. They never execute the supplied command.

> **Operating status:** v1 is advisory-only. A `safe` result is not permission to execute a command. See the [model card](docs/model-card.md) and [release criteria](docs/release-criteria.md).

## Quick start

Requirements: Python 3.11+, Go 1.25+, and `make`.

```bash
make setup
make quality
make test
make corpus-report
make train
```

`make train` validates the reviewed evaluation manifest and rejects development/evaluation overlap before it writes an artifact.

## Synthetic data workflow

The development corpus is generated in focused batches using `std_slop` and the task prompt in `prompts/command-safety-v1.md`.

```bash
make generate BATCH=013 FOCUS='quoted destructive commands and safe near-neighbours'
```

Generation output is raw input, not training-ready data. Validate, review, and merge batches before training:

```bash
make validate-data DATASET=data/command-safety/raw/std-slop-batch-013.jsonl
python -m commandclassifier.merge_data \
  --input data/command-safety/raw/*.jsonl \
  --output data/command-safety/processed/development.jsonl
make corpus-report
```

Synthetic data is useful for development coverage but cannot authorize enforcement. Preserve prompts, generator details, review decisions, hashes, and manifests. For independently sourced evaluation evidence and maintenance requirements, see [external evaluation evidence](docs/external-evaluation-evidence.md).

## Python inference

Train first if `artifacts/command-safety-v1/` does not contain a current artifact:

```bash
make train
make infer TEXT='git status'
make infer TEXT='rm -rf /tmp/example'
```

The Python CLI verifies the portable bundle hash and Python model hash. It emits JSON with `label`, `unsafe_probability`, `confidence`, `review_recommended`, and `model_version`.

## Go inference

The Go runtime reads the portable `model.json` and its required `model.json.sha256` sidecar. It does not require Python or `model.joblib` at inference time.

For development use:

```bash
make train
make go-infer TEXT='git status'
make go-infer TEXT='git reset --hard HEAD~3'
```

To install the Go binary locally:

```bash
make go-install
./bin/command-classify \
  --model artifacts/command-safety-v1/model.json \
  --text 'git status'
```

To deploy it separately, copy both portable artifact files with the binary:

```text
command-classify
model.json
model.json.sha256
```

The portable JSON contains the model weights: vocabularies, IDF values, linear coefficients, intercept, calibration mapping, normalization metadata, and decision policy. The Python-only `model.joblib` is not required by Go. Keep the bundle and sidecar together; Go fails if the sidecar hash does not match.

Test the Go implementation with:

```bash
make go-test
cd go && go test -run=^$ \
  -fuzz=FuzzClassifyNeverReturnsInvalidProbability \
  -fuzztime=5s ./cmd/command-classify
```

## Creating another task

The data contract is reusable: create a new task manifest, prompt, raw/processed/reviewed dataset directories, and a locked reviewed evaluation manifest. Keep labels mutually exclusive, record provenance and review metadata, and require overlap checks before training.

The current trainer and portable artifact are intentionally specialized to binary `safe`/`unsafe` scoring: they name `unsafe` as the positive class and emit command-safety policy fields. To make a fully generic task:

1. Add task-configured labels, positive class, model version, and decision-policy output fields.
2. Replace the fixed `safe`/`unsafe` checks in `train.py`, `cli.py`, exporter, and Go runtime with validated manifest configuration.
3. Define task-specific prompt, schema extensions, class balance, grouped split fields, and review policy.
4. Add a task-specific locked evaluation corpus and model card.
5. Add Python/Go fixtures for the new artifact configuration before production use.

Do not weaken the review, provenance, overlap, calibration, or release-evidence gates when adding a task.

## Useful commands

```bash
make help
make quality
make test
make corpus-report
make train
make infer TEXT='git status'
make go-infer TEXT='git status'
make go-install
make go-test
```
